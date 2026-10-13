"""Core services for the Cedalion DOT Web application.

The module deliberately keeps Streamlit out of the computation layer so that file
parsing, registration, forward modelling, and reconstruction can be tested and reused
by a future API server.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr


LANDMARK_ALIASES = {
    "NZHS": "Nz",
    "NASION": "Nz",
    "NAS": "Nz",
    "NZ": "Nz",
    "IZHS": "Iz",
    "INION": "Iz",
    "IZ": "Iz",
    "ARHS": "RPA",
    "RIGHTPREAURICULAR": "RPA",
    "RPA": "RPA",
    "ALHS": "LPA",
    "LEFTPREAURICULAR": "LPA",
    "LPA": "LPA",
    "CZHS": "Cz",
    "CZ": "Cz",
}

POINT_COLUMN_ALIASES = {
    "label": {"label", "name", "point", "pointname", "id", "optode"},
    "x": {"x", "posx", "coordx", "xmm"},
    "y": {"y", "posy", "coordy", "ymm"},
    "z": {"z", "posz", "coordz", "zmm"},
    "type": {"type", "kind", "pointtype", "optodetype"},
}

MEASUREMENT_COLUMN_ALIASES = {
    "channel": {"channel", "channelname", "ch", "measurement"},
    "source": {"source", "src", "sourceindex", "s"},
    "detector": {"detector", "det", "detectorindex", "d"},
    "wavelength": {"wavelength", "wl", "lambda", "wavelengthnm"},
}

REQUIRED_SAVED_HEAD_FILES = {
    "segmentation_masks.nc",
    "brain.ply",
    "scalp.ply",
    "t_ijk2ras.nc",
    "voxel_to_vertex_brain.npz",
    "voxel_to_vertex_scalp.npz",
}

DEFAULT_MASK_FILES = {
    "csf": "mask_csf.nii",
    "gm": "mask_gray.nii",
    "scalp": "mask_skin.nii",
    "skull": "mask_bone.nii",
    "wm": "mask_white.nii",
}


@dataclass
class ImportBundle:
    """Data imported from one SNIRF file."""

    recording: Any
    geo3d: Any
    measurement_list: pd.DataFrame
    data_keys: list[str]
    warnings: list[str] = field(default_factory=list)


@dataclass
class RegistrationResult:
    """Registered geometry plus exportable registration diagnostics."""

    registered_geo3d: Any
    head_ras: Any
    parameters: dict[str, Any]
    registered_coordinates: pd.DataFrame


@dataclass
class ForwardResult:
    """Files and arrays produced by a forward-model run."""

    adot: xr.DataArray
    fluence_path: Path
    adot_path: Path
    log_path: Path
    solver: str
    parameters: dict[str, Any]


def _import_cedalion():
    """Import Cedalion lazily so table-only helpers remain lightweight."""

    import cedalion
    import cedalion.dot
    import cedalion.geometry.registration
    import cedalion.io
    import cedalion.nirs

    return cedalion


def canonical_token(value: Any) -> str:
    """Return a comparison token insensitive to case and punctuation."""

    return re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())


def normalize_label(value: Any) -> str:
    """Normalize common anatomical labels while preserving optode identifiers."""

    raw = str(value).strip()
    return LANDMARK_ALIASES.get(canonical_token(raw), raw)


def _rename_columns(frame: pd.DataFrame, aliases: dict[str, set[str]]) -> pd.DataFrame:
    lookup: dict[str, str] = {}
    for canonical, names in aliases.items():
        for name in names | {canonical}:
            lookup[canonical_token(name)] = canonical

    renamed: dict[Any, str] = {}
    for column in frame.columns:
        token = canonical_token(column)
        if token in lookup:
            renamed[column] = lookup[token]
    return frame.rename(columns=renamed)


def read_table_bytes(filename: str, payload: bytes) -> pd.DataFrame:
    """Read a CSV, TSV, JSON, or Excel table from uploaded bytes."""

    suffix = Path(filename).suffix.lower()
    buffer = io.BytesIO(payload)
    if suffix == ".csv":
        return pd.read_csv(buffer)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(buffer, sep="\t")
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(buffer)
    if suffix == ".json":
        value = json.loads(payload.decode("utf-8-sig"))
        if isinstance(value, dict):
            for key in ("points", "coordinates", "data", "measurements"):
                if key in value and isinstance(value[key], list):
                    value = value[key]
                    break
        return pd.DataFrame(value)
    raise ValueError(f"Unsupported table format: {suffix or '<no suffix>'}")


def _point_type_from_value(value: Any, label: str):
    cedalion = _import_cedalion()
    point_type = cedalion.dataclasses.PointType
    token = canonical_token(value) if value is not None and not pd.isna(value) else ""
    if token in {"SOURCE", "SRC", "S", "1"}:
        return point_type.SOURCE
    if token in {"DETECTOR", "DET", "D", "2"}:
        return point_type.DETECTOR
    if token in {"LANDMARK", "FIDUCIAL", "ANATOMICAL", "LM", "3"}:
        return point_type.LANDMARK
    if token in {"ELECTRODE", "EEG", "4"}:
        return point_type.ELECTRODE

    if re.match(r"^S\d", label, flags=re.IGNORECASE):
        return point_type.SOURCE
    if re.match(r"^D\d", label, flags=re.IGNORECASE):
        return point_type.DETECTOR
    if normalize_label(label) in {"Nz", "Iz", "LPA", "RPA", "Cz"}:
        return point_type.LANDMARK
    return point_type.UNKNOWN


def normalize_point_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an optode/anatomical point table."""

    frame = _rename_columns(frame.copy(), POINT_COLUMN_ALIASES)
    required = {"label", "x", "y", "z"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Coordinate table is missing columns: " + ", ".join(missing))

    columns = ["label", "x", "y", "z"] + (["type"] if "type" in frame else [])
    frame = frame[columns].dropna(subset=["label", "x", "y", "z"]).copy()
    frame["label"] = frame["label"].map(normalize_label)
    for axis in ("x", "y", "z"):
        frame[axis] = pd.to_numeric(frame[axis], errors="raise")

    if frame["label"].duplicated().any():
        duplicate = frame.loc[frame["label"].duplicated(), "label"].tolist()
        raise ValueError(f"Coordinate labels must be unique; duplicates: {duplicate}")
    if not np.isfinite(frame[["x", "y", "z"]].to_numpy(float)).all():
        raise ValueError("Coordinate table contains non-finite values")

    raw_types = frame["type"] if "type" in frame else [None] * len(frame)
    frame["type"] = [
        _point_type_from_value(value, label)
        for value, label in zip(raw_types, frame["label"], strict=True)
    ]
    return frame.reset_index(drop=True)


def build_labeled_points(
    frame: pd.DataFrame, *, units: str = "mm", crs: str = "digitized"
):
    """Build a Cedalion LabeledPoints object from a normalized point table."""

    cedalion = _import_cedalion()
    normalized = normalize_point_table(frame)
    return cedalion.dataclasses.build_labeled_points(
        normalized[["x", "y", "z"]].to_numpy(float),
        crs=crs,
        units=units,
        labels=normalized["label"].astype(str).tolist(),
        types=normalized["type"].tolist(),
    )


def _resolve_index_label(prefix: str, value: Any, available: set[str] | None) -> str:
    raw = str(value).strip()
    if not re.fullmatch(r"\d+(?:\.0+)?", raw):
        return raw
    number = int(float(raw))
    candidates = [f"{prefix}{number}", f"{prefix}{number:02d}", f"{prefix}{number:03d}"]
    if available:
        for candidate in candidates:
            if candidate in available:
                return candidate
    return candidates[0]


def normalize_measurement_list(
    frame: pd.DataFrame, geo_labels: Iterable[str] | None = None
) -> pd.DataFrame:
    """Normalize a measurement list to Cedalion's four-column contract."""

    frame = _rename_columns(frame.copy(), MEASUREMENT_COLUMN_ALIASES)
    required = {"source", "detector", "wavelength"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Measurement list is missing columns: " + ", ".join(missing))

    available = set(map(str, geo_labels)) if geo_labels is not None else None
    frame = frame.dropna(subset=["source", "detector", "wavelength"]).copy()
    frame["source"] = [
        _resolve_index_label("S", value, available) for value in frame["source"]
    ]
    frame["detector"] = [
        _resolve_index_label("D", value, available) for value in frame["detector"]
    ]
    frame["wavelength"] = pd.to_numeric(frame["wavelength"], errors="raise").astype(
        float
    )

    if "channel" not in frame:
        frame["channel"] = frame["source"].astype(str) + frame["detector"].astype(str)
    frame["channel"] = frame["channel"].astype(str).str.strip()

    normalized = frame[
        ["channel", "source", "detector", "wavelength"]
    ].drop_duplicates()
    pairs_per_channel = normalized.groupby("channel")[["source", "detector"]].nunique()
    invalid = pairs_per_channel[(pairs_per_channel > 1).any(axis=1)].index.tolist()
    if invalid:
        raise ValueError(
            f"Each channel must map to one source-detector pair: {invalid}"
        )

    if available is not None:
        used = set(normalized["source"]) | set(normalized["detector"])
        missing_labels = sorted(used - available)
        if missing_labels:
            raise ValueError(
                "Measurement list references coordinates that are not loaded: "
                + ", ".join(missing_labels)
            )

    return normalized.sort_values(["channel", "wavelength"]).reset_index(drop=True)


def measurement_list_from_timeseries(timeseries: xr.DataArray) -> pd.DataFrame:
    """Derive source-detector-wavelength rows from a Cedalion time series."""

    required_coords = {"channel", "source", "detector", "wavelength"}
    missing = sorted(required_coords - set(timeseries.coords))
    if missing:
        raise ValueError("Time series lacks coordinates: " + ", ".join(missing))

    rows: list[dict[str, Any]] = []
    channels = timeseries.coords["channel"].values
    sources = timeseries.coords["source"].values
    detectors = timeseries.coords["detector"].values
    wavelengths = timeseries.coords["wavelength"].values
    for channel, source, detector in zip(channels, sources, detectors, strict=True):
        for wavelength in wavelengths:
            rows.append(
                {
                    "channel": str(channel),
                    "source": str(source),
                    "detector": str(detector),
                    "wavelength": float(wavelength),
                }
            )
    return normalize_measurement_list(rows_to_frame(rows))


def rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Small named helper used to make table provenance explicit in tests."""

    return pd.DataFrame(rows)


def _rename_geo_labels(geo3d):
    labels = [normalize_label(value) for value in geo3d.label.values]
    if len(labels) != len(set(labels)):
        raise ValueError("Landmark alias normalization created duplicate labels")
    return geo3d.assign_coords(label=("label", labels))


def save_uploaded_file(filename: str, payload: bytes, directory: Path) -> Path:
    """Store uploaded bytes under a controlled session directory."""

    directory.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    path = directory / safe_name
    path.write_bytes(payload)
    return path


def read_snirf(path: str | Path) -> ImportBundle:
    """Load the first NIRS element from a SNIRF file."""

    cedalion = _import_cedalion()
    recordings = cedalion.io.read_snirf(path)
    if not recordings:
        raise ValueError("SNIRF file contains no NIRS recording")
    recording = recordings[0]
    geo3d = _rename_geo_labels(recording.geo3d)
    recording.geo3d = geo3d

    data_keys = list(recording.timeseries.keys())
    candidates = [key for key in ("od", "amp") if key in data_keys] + data_keys
    warnings: list[str] = []
    measurement_list = None
    for key in dict.fromkeys(candidates):
        try:
            measurement_list = measurement_list_from_timeseries(recording[key])
            break
        except ValueError:
            continue
    if measurement_list is None:
        raise ValueError(
            "No SNIRF time series contains the required channel, source, "
            "detector, and wavelength coordinates"
        )
    if len(recordings) > 1:
        warnings.append(
            "The SNIRF file has multiple NIRS elements; the web MVP loaded "
            "the first one."
        )
    return ImportBundle(recording, geo3d, measurement_list, data_keys, warnings)


def safe_extract_zip(payload: bytes, destination: Path) -> Path:
    """Extract a ZIP upload while rejecting path traversal entries."""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(destination)
    return destination


def _find_directory_with_files(root: Path, filenames: set[str]) -> Path | None:
    candidates = [root] + [path for path in root.rglob("*") if path.is_dir()]
    for directory in candidates:
        present = {item.name for item in directory.iterdir() if item.is_file()}
        if filenames <= present:
            return directory
    return None


def load_custom_head_model_zip(payload: bytes, workspace: Path):
    """Load a Cedalion saved-head-model ZIP or a segmentation/surface ZIP."""

    cedalion = _import_cedalion()
    extract_root = workspace / "custom_head_model"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    safe_extract_zip(payload, extract_root)

    saved_dir = _find_directory_with_files(extract_root, REQUIRED_SAVED_HEAD_FILES)
    if saved_dir is not None:
        return cedalion.dot.TwoSurfaceHeadModel.load(saved_dir)

    config_path = next(iter(extract_root.rglob("head_model.json")), None)
    config: dict[str, Any] = {}
    if config_path is not None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        base_dir = config_path.parent
    else:
        base_dir = extract_root

    mask_files = config.get("mask_files", DEFAULT_MASK_FILES)
    segmentation_dir = _find_directory_with_files(base_dir, set(mask_files.values()))
    if segmentation_dir is None:
        raise ValueError(
            "ZIP is neither a saved TwoSurfaceHeadModel nor a segmentation bundle "
            f"containing {sorted(mask_files.values())}"
        )

    def resolve_optional(config_key: str, candidates: tuple[str, ...]) -> Path | None:
        configured = config.get(config_key)
        if configured:
            path = segmentation_dir / configured
            return path if path.exists() else None
        for name in candidates:
            found = next(iter(segmentation_dir.rglob(name)), None)
            if found is not None:
                return found
        return None

    landmarks = resolve_optional("landmarks_ras_file", ("landmarks.mrk.json",))
    brain_surface = resolve_optional(
        "brain_surface_file", ("mask_brain.obj", "brain.obj", "brain.ply")
    )
    scalp_surface = resolve_optional(
        "scalp_surface_file", ("mask_scalp.obj", "scalp.obj", "scalp.ply")
    )
    coordinates = resolve_optional(
        "coordinates_file", ("brain_vertex_coordinates.csv",)
    )

    common = dict(
        segmentation_dir=str(segmentation_dir),
        mask_files=mask_files,
        landmarks_ras_file=str(landmarks) if landmarks else None,
        brain_face_count=config.get("brain_face_count"),
        scalp_face_count=config.get("scalp_face_count"),
        smoothing=float(config.get("smoothing", 0.0)),
    )
    if brain_surface is not None and scalp_surface is not None:
        return cedalion.dot.TwoSurfaceHeadModel.from_surfaces(
            **common,
            brain_surface_file=str(brain_surface),
            scalp_surface_file=str(scalp_surface),
            coordinates_file=str(coordinates) if coordinates else None,
        )
    return cedalion.dot.TwoSurfaceHeadModel.from_segmentation(**common)


def load_standard_head_model(name: str):
    """Load a cached Cedalion standard head model."""

    cedalion = _import_cedalion()
    return cedalion.dot.get_standard_headmodel(name)


def head_to_ras(head_model):
    """Return a head model in its scanner/RAS coordinate system."""

    if head_model.crs == "ijk":
        return head_model.apply_transform(head_model.t_ijk2ras)
    return head_model


def _initial_transform(head_ras, geo3d, mode: str):
    cedalion = _import_cedalion()
    registration = cedalion.geometry.registration
    functions = {
        "trans_rot_isoscale": registration.register_trans_rot_isoscale,
        "trans_rot": registration.register_trans_rot,
        "general": registration.register_general_affine,
        "identity": registration.register_identity,
    }
    try:
        function = functions[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown registration mode: {mode}") from exc
    return function(head_ras.landmarks, geo3d)


def _points_to_frame(points) -> pd.DataFrame:
    values = points.pint.to("mm").pint.dequantify().values
    return pd.DataFrame(
        {
            "label": points.label.values.astype(str),
            "type": [
                getattr(value, "name", str(value)) for value in points.type.values
            ],
            "x_mm": values[:, 0],
            "y_mm": values[:, 1],
            "z_mm": values[:, 2],
        }
    )


def register_geometry(
    head_model,
    geo3d,
    measurement_list: pd.DataFrame | None,
    *,
    mode: str = "trans_rot_isoscale",
    use_spring_icp: bool = True,
    n_iter: int = 400,
    k_spring: float = 1.0,
    k_anchor: float = 10.0,
    step_size: float = 0.1,
    convergence_tol: float = 0.01,
) -> RegistrationResult:
    """Register digitized geometry to a head model and collect QC metrics."""

    head_ras = head_to_ras(head_model)
    if head_ras.landmarks is None:
        raise ValueError("The selected head model has no anatomical landmarks")

    geo3d = _rename_geo_labels(geo3d)
    common = head_ras.landmarks.points.common_labels(geo3d)
    minimum = 4 if mode == "general" else 3
    if mode != "identity" and len(common) < minimum:
        raise ValueError(
            f"Registration mode {mode!r} needs at least {minimum} common landmarks; "
            f"found {list(common)}"
        )

    transform = _initial_transform(head_ras, geo3d, mode)
    initially_aligned = geo3d.points.apply_transform(transform)

    details = None
    if use_spring_icp:
        if measurement_list is None or measurement_list.empty:
            raise ValueError("Spring-ICP registration requires a measurement list")
        registered, details = head_ras.align_and_relax_to_scalp(
            geo3d,
            measurement_list,
            n_iter=int(n_iter),
            k_spring=float(k_spring),
            k_anchor=float(k_anchor),
            step_size=float(step_size),
            convergence_tol=float(convergence_tol),
            initial_align_mode=mode,
        )
    else:
        registered = head_ras.scalp.snap(initially_aligned)

    matrix = np.asarray(transform.pint.dequantify()).tolist()
    target = head_ras.landmarks.sel(label=common).pint.to("mm").pint.dequantify()
    initial_common = initially_aligned.sel(label=common).pint.to("mm").pint.dequantify()
    final_common = registered.sel(label=common).pint.to("mm").pint.dequantify()
    initial_errors = np.linalg.norm(initial_common.values - target.values, axis=1)
    final_errors = np.linalg.norm(final_common.values - target.values, axis=1)

    parameters: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "spring_icp" if use_spring_icp else "landmark_align_and_snap",
        "initial_align_mode": mode,
        "source_crs": geo3d.points.crs,
        "target_crs": head_ras.crs,
        "common_landmarks": list(map(str, common)),
        "initial_transform_matrix": matrix,
        "initial_landmark_errors_mm": dict(
            zip(map(str, common), initial_errors.tolist())
        ),
        "final_landmark_errors_mm": dict(zip(map(str, common), final_errors.tolist())),
        "initial_landmark_rmse_mm": float(np.sqrt(np.mean(initial_errors**2))),
        "final_landmark_rmse_mm": float(np.sqrt(np.mean(final_errors**2))),
        "note": (
            "initial_transform_matrix describes only the global phase. Final "
            "spring-ICP coordinates are non-rigid and are saved separately."
        ),
    }
    if details is not None:
        spring_errors = np.asarray(details.spring_errors.pint.dequantify())
        parameters["spring_icp"] = {
            "n_iterations": int(details.n_iterations),
            "converged": bool(details.converged),
            "max_abs_spring_error": float(np.max(np.abs(spring_errors)))
            if spring_errors.size
            else None,
            "median_abs_spring_error": float(np.median(np.abs(spring_errors)))
            if spring_errors.size
            else None,
            "max_snap_displacement_per_iteration": np.asarray(
                details.snap_displacement_per_iter
            )
            .astype(float)
            .tolist(),
        }

    return RegistrationResult(
        registered_geo3d=registered,
        head_ras=head_ras,
        parameters=parameters,
        registered_coordinates=_points_to_frame(registered),
    )


@lru_cache(maxsize=None)
def _probe_mcx_runtime(module_name: str) -> tuple[bool, str]:
    """Run a tiny isolated simulation to verify an MCX backend and its driver."""

    if importlib.util.find_spec(module_name) is None:
        return False, "Python package is not installed"
    probe_script = f"""
import importlib
import json
import numpy as np

module = importlib.import_module({module_name!r})
devices = module.gpuinfo()
config = {{
    "nphoton": 1000,
    "vol": np.ones((8, 8, 8), dtype=np.uint8),
    "tstart": 0,
    "tend": 5e-9,
    "tstep": 5e-9,
    "srcpos": [4, 4, 1],
    "srcdir": [0, 0, 1],
    "prop": [[0, 0, 1, 1], [0.01, 1, 0.9, 1.37]],
    "issrcfrom0": 1,
    "isnormalized": 1,
    "outputtype": "fluence",
    "issavedet": 0,
    "unitinmm": 1,
}}
result = module.run(config)
flux = np.asarray(result["flux"])
if flux.size == 0 or not np.isfinite(flux).all():
    raise RuntimeError("runtime probe produced an invalid fluence array")
names = [str(device.get("name", "unknown device")) for device in devices]
print("__CEDALION_DOT_PROBE__" + json.dumps(names))
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe_script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        marker = "__CEDALION_DOT_PROBE__"
        combined = completed.stdout + "\n" + completed.stderr
        if completed.returncode != 0 or marker not in combined:
            detail = (
                combined.strip().splitlines()[-1]
                if combined.strip()
                else "unknown error"
            )
            return False, detail
        names = json.loads(combined.split(marker, 1)[1].splitlines()[0])
        return True, ", ".join(names)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def solver_capabilities(
    repo_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Report availability of Cedalion's supported photon solvers."""

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    plugin_path = root / "plugins" / "nirfaster-uFF" / "nirfasteruff"
    cuda_available, cuda_detail = _probe_mcx_runtime("pmcx")
    opencl_available, opencl_detail = _probe_mcx_runtime("pmcxcl")
    nirfaster_installed = (
        plugin_path.exists() or importlib.util.find_spec("nirfasteruff") is not None
    )
    return {
        "mcx_cuda": {
            "available": cuda_available,
            "label": "MCX (CUDA/GPU)",
            "requirement": "pmcx and a CUDA-capable GPU",
            "detail": cuda_detail,
        },
        "mcx_opencl": {
            "available": opencl_available,
            "label": "MCXCL (OpenCL/CPU or GPU)",
            "requirement": "pmcxcl and a working OpenCL runtime",
            "detail": opencl_detail,
        },
        "nirfaster": {
            "available": nirfaster_installed,
            "label": "NIRFASTer-uFF",
            "requirement": (
                "Cedalion plugins/nirfaster-uFF installation matching Python"
            ),
            "detail": (
                "Plugin path found; runtime is checked when the job starts"
                if nirfaster_installed
                else "Plugin is not installed"
            ),
        },
    }


def validate_forward_inputs(geo3d, measurement_list: pd.DataFrame) -> pd.DataFrame:
    """Validate the real measurement-pair contract before photon simulation."""

    normalized = normalize_measurement_list(measurement_list, geo3d.label.values)
    point_types = dict(
        zip(geo3d.label.values.astype(str), geo3d.type.values, strict=True)
    )
    cedalion = _import_cedalion()
    for source in normalized["source"].unique():
        if point_types[source] != cedalion.dataclasses.PointType.SOURCE:
            raise ValueError(
                f"{source} is referenced as a source but its point type differs"
            )
    for detector in normalized["detector"].unique():
        if point_types[detector] != cedalion.dataclasses.PointType.DETECTOR:
            raise ValueError(
                f"{detector} is referenced as a detector but its point type differs"
            )
    return normalized


def select_used_optodes(geo3d, measurement_list: pd.DataFrame):
    """Remove unused optodes so a forward run does not simulate unmeasured pairs."""

    labels = set(measurement_list["source"]) | set(measurement_list["detector"])
    mask = np.array([str(label) in labels for label in geo3d.label.values])
    return geo3d.sel(label=mask)


def run_forward_model(
    head_model,
    registered_geo3d,
    measurement_list: pd.DataFrame,
    *,
    solver: str,
    output_dir: Path,
    nphoton: int = 10_000_000,
) -> ForwardResult:
    """Run MCX/MCXCL/NIRFASTer and calculate the Adot sensitivity matrix."""

    cedalion = _import_cedalion()
    output_dir.mkdir(parents=True, exist_ok=True)
    measurement_list = validate_forward_inputs(registered_geo3d, measurement_list)
    used_geo3d = select_used_optodes(registered_geo3d, measurement_list)

    fluence_path = output_dir / "fluence.h5"
    adot_path = output_dir / "Adot.nc"
    log_path = output_dir / "forward_model.log"
    model = cedalion.dot.ForwardModel(
        head_model,
        used_geo3d,
        measurement_list,
        log_file=log_path,
    )
    if solver == "mcx_cuda":
        model.compute_fluence_mcx(fluence_path, nphoton=int(nphoton), cuda=True)
    elif solver == "mcx_opencl":
        model.compute_fluence_mcx(fluence_path, nphoton=int(nphoton), cuda=False)
    elif solver == "nirfaster":
        model.compute_fluence_nirfaster(fluence_path)
    else:
        raise ValueError(f"Unknown forward solver: {solver}")
    model.compute_sensitivity(fluence_path, adot_path)
    adot = cedalion.io.load_Adot(adot_path)
    voxel_dimensions = model._get_voxel_dimensions().to("mm").magnitude
    parameters = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "solver": solver,
        "nphoton": int(nphoton) if solver.startswith("mcx") else None,
        "channel_count": int(measurement_list.channel.nunique()),
        "measurement_row_count": int(len(measurement_list)),
        "used_optode_count": int(len(used_geo3d.label)),
        "wavelengths_nm": sorted(map(float, measurement_list.wavelength.unique())),
        "head_model_crs": str(head_model.crs),
        "voxel_dimensions_mm": np.asarray(voxel_dimensions, dtype=float).tolist(),
        "fluence_file": fluence_path.name,
        "adot_file": adot_path.name,
        "adot_dimensions": dict(adot.sizes),
        "known_limitations": [
            "Cedalion currently reuses one MCX fluence simulation across wavelengths.",
            "The NIRFASTer path currently builds tissue properties from "
            "wavelength index 0.",
            "Adot normalization for non-1-mm voxels requires independent validation.",
        ],
    }
    return ForwardResult(
        adot,
        fluence_path,
        adot_path,
        log_path,
        solver,
        parameters,
    )


def load_adot_bytes(payload: bytes, directory: Path) -> xr.DataArray:
    """Load a previously saved Cedalion Adot NetCDF upload."""

    path = save_uploaded_file("uploaded_Adot.nc", payload, directory)
    adot = xr.load_dataarray(path)
    required = {"channel", "vertex", "wavelength"}
    if not required <= set(adot.dims):
        raise ValueError(f"Adot must contain dimensions {sorted(required)}")
    return adot


def prepare_optical_density(recording, data_key: str) -> xr.DataArray:
    """Return OD data, converting raw amplitude when explicitly selected."""

    cedalion = _import_cedalion()
    timeseries = recording[data_key]
    if data_key.lower() in {"amp", "amplitude", "intensity", "raw"}:
        return cedalion.nirs.cw.int2od(timeseries)
    return timeseries


def align_data_to_adot(data: xr.DataArray, adot: xr.DataArray) -> xr.DataArray:
    """Select and order measurement data to match Adot's channel/wavelength axes."""

    for dim in ("channel", "wavelength"):
        if dim not in data.dims:
            raise ValueError(f"Measurement data lacks required dimension {dim!r}")
    missing_channels = sorted(
        set(map(str, adot.channel.values)) - set(map(str, data.channel.values))
    )
    if missing_channels:
        raise ValueError(
            "Data is missing Adot channels: " + ", ".join(missing_channels)
        )
    missing_wavelengths = sorted(
        set(map(float, adot.wavelength.values))
        - set(map(float, data.wavelength.values))
    )
    if missing_wavelengths:
        raise ValueError(
            "Data is missing Adot wavelengths: "
            + ", ".join(map(str, missing_wavelengths))
        )
    return data.sel(channel=adot.channel.values, wavelength=adot.wavelength.values)


def run_reconstruction(
    adot: xr.DataArray,
    measurement_data: xr.DataArray,
    *,
    recon_mode: str = "mua2conc",
    brain_only: bool = True,
    alpha_meas: float = 0.01,
    alpha_spatial: float | None = None,
    lambda_r_conc: float | None = None,
) -> xr.DataArray:
    """Run Cedalion's regularized DOT inverse reconstruction."""

    cedalion = _import_cedalion()
    measurement_data = align_data_to_adot(measurement_data, adot)
    reconstructor = cedalion.dot.ImageRecon(
        adot,
        recon_mode=recon_mode,
        brain_only=brain_only,
        alpha_meas=float(alpha_meas),
        alpha_spatial=alpha_spatial,
        lambda_R_conc=lambda_r_conc,
        apply_c_meas=False,
        spatial_basis_functions=None,
    )
    return reconstructor.reconstruct(measurement_data)


def dataarray_to_netcdf_bytes(data: xr.DataArray) -> bytes:
    """Serialize an xarray DataArray for a browser download."""

    return data.pint.dequantify().to_netcdf()


def dataarray_to_npz_bytes(data: xr.DataArray) -> bytes:
    """Serialize values and simple coordinates to a compressed NumPy archive."""

    data = data.pint.dequantify()
    buffer = io.BytesIO()
    payload: dict[str, Any] = {"values": data.values, "dims": np.array(data.dims)}
    for name, coord in data.coords.items():
        if coord.ndim == 1:
            payload[f"coord_{name}"] = coord.values
    np.savez_compressed(buffer, **payload)
    return buffer.getvalue()


def json_bytes(value: dict[str, Any]) -> bytes:
    """Serialize a JSON-safe mapping with stable formatting."""

    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


def make_project_manifest(
    *,
    coordinate_source: str | None,
    measurement_source: str | None,
    head_model_source: str | None,
    registration: dict[str, Any] | None,
    forward: ForwardResult | None,
    reconstruction: xr.DataArray | None,
) -> dict[str, Any]:
    """Build a compact, auditable manifest for the current Web project."""

    manifest = {
        "format": "cedalion-dot-web-project",
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "coordinates": coordinate_source,
            "measurement_list": measurement_source,
            "head_model": head_model_source,
        },
        "registration": registration,
        "forward_model": None,
        "reconstruction": None,
    }
    if forward is not None:
        manifest["forward_model"] = forward.parameters
    if reconstruction is not None:
        manifest["reconstruction"] = {
            "dims": dict(reconstruction.sizes),
            "units": str(reconstruction.pint.units),
        }
    return manifest


def make_session_workspace() -> Path:
    """Create a private temporary workspace for a browser session."""

    return Path(tempfile.mkdtemp(prefix="cedalion_dot_web_"))
