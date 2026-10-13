from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

import cedalion.data
import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from apps.dot_web import core


def test_read_snirf_accepts_unicode_subject_id(tmp_path: Path):
    source = cedalion.data.get_fingertapping_snirf_path()
    path = tmp_path / "unicode_subject.snirf"
    shutil.copyfile(source, path)
    subject_id = "受试者001"

    with h5py.File(path, "r+") as snirf_file:
        metadata = snirf_file["nirs/metaDataTags"]
        del metadata["SubjectID"]
        metadata.create_dataset(
            "SubjectID",
            data=subject_id,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    bundle = core.read_snirf(path)

    assert bundle.recording.meta_data["SubjectID"] == subject_id


def test_normalize_point_table_maps_landmarks_and_types():
    frame = pd.DataFrame(
        {
            "Label": ["NzHS", "S01", "D01"],
            "X": [0.0, 1.0, 2.0],
            "Y": [0.0, 1.0, 2.0],
            "Z": [0.0, 1.0, 2.0],
        }
    )

    normalized = core.normalize_point_table(frame)

    assert normalized.label.tolist() == ["Nz", "S01", "D01"]
    assert [value.name for value in normalized.type] == [
        "LANDMARK",
        "SOURCE",
        "DETECTOR",
    ]


def test_normalize_measurement_list_resolves_numeric_indices():
    frame = pd.DataFrame(
        {
            "SourceIndex": [1, 1],
            "DetectorIndex": [1, 1],
            "Wavelength": [760, 850],
        }
    )

    normalized = core.normalize_measurement_list(frame, ["S01", "D01"])

    assert normalized.channel.tolist() == ["S01D01", "S01D01"]
    assert normalized.source.tolist() == ["S01", "S01"]
    assert normalized.detector.tolist() == ["D01", "D01"]


def test_normalize_measurement_list_rejects_unknown_optodes():
    frame = pd.DataFrame(
        {
            "channel": ["ch1"],
            "source": ["S02"],
            "detector": ["D01"],
            "wavelength": [760],
        }
    )

    with pytest.raises(ValueError, match="not loaded"):
        core.normalize_measurement_list(frame, ["S01", "D01"])


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../escape.txt", "unsafe")

    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        core.safe_extract_zip(buffer.getvalue(), tmp_path / "extract")


def test_align_data_to_adot_restores_channel_and_wavelength_order():
    adot = xr.DataArray(
        np.ones((2, 3, 2)),
        dims=["channel", "vertex", "wavelength"],
        coords={
            "channel": ["ch1", "ch2"],
            "wavelength": [760.0, 850.0],
            "is_brain": ("vertex", [True, True, False]),
        },
    )
    data = xr.DataArray(
        np.arange(8).reshape(2, 2, 2),
        dims=["channel", "wavelength", "time"],
        coords={
            "channel": ["ch2", "ch1"],
            "wavelength": [850.0, 760.0],
            "time": [0.0, 1.0],
        },
    )

    aligned = core.align_data_to_adot(data, adot)

    assert aligned.channel.values.tolist() == ["ch1", "ch2"]
    assert aligned.wavelength.values.tolist() == [760.0, 850.0]


def test_run_reconstruction_with_synthetic_adot():
    rng = np.random.default_rng(42)
    channels = [f"ch{i}" for i in range(4)]
    wavelengths = [760.0, 850.0]
    adot = xr.DataArray(
        rng.uniform(0.1, 1.0, size=(4, 6, 2)),
        dims=["channel", "vertex", "wavelength"],
        coords={
            "channel": channels,
            "source": ("channel", ["S1", "S2", "S3", "S4"]),
            "detector": ("channel", ["D1", "D2", "D3", "D4"]),
            "wavelength": wavelengths,
            "is_brain": ("vertex", [True, True, True, True, False, False]),
        },
    )
    data = xr.DataArray(
        rng.normal(size=(4, 2, 3)),
        dims=["channel", "wavelength", "time"],
        coords={
            "channel": channels,
            "wavelength": wavelengths,
            "time": [0.0, 1.0, 2.0],
        },
    )

    image = core.run_reconstruction(
        adot,
        data,
        recon_mode="mua",
        brain_only=True,
        alpha_meas=0.01,
    )

    assert image.sizes["vertex"] == 4
    assert image.sizes["wavelength"] == 2
    assert image.sizes["time"] == 3
    assert np.isfinite(image.pint.dequantify()).all()


@pytest.mark.skipif(
    not core.solver_capabilities()["mcx_opencl"]["available"],
    reason="MCXCL runtime is unavailable",
)
def test_run_forward_model_with_tiny_head(tmp_path: Path):
    cedalion = core._import_cedalion()
    shape = (18, 18, 18)
    grid = np.indices(shape).transpose(1, 2, 3, 0)
    radius = np.linalg.norm(grid - np.array([8.5, 8.5, 8.5]), axis=-1)
    masks = {
        "wm": radius <= 3.0,
        "gm": (radius > 3.0) & (radius <= 4.0),
        "csf": (radius > 4.0) & (radius <= 5.0),
        "skull": (radius > 5.0) & (radius <= 6.0),
        "scalp": (radius > 6.0) & (radius <= 7.5),
    }
    filenames = {
        "csf": "mask_csf.nii",
        "gm": "mask_gray.nii",
        "scalp": "mask_skin.nii",
        "skull": "mask_bone.nii",
        "wm": "mask_white.nii",
    }
    for tissue, filename in filenames.items():
        nib.save(
            nib.Nifti1Image(masks[tissue].astype(np.uint8), np.eye(4)),
            tmp_path / filename,
        )

    head = cedalion.dot.TwoSurfaceHeadModel.from_segmentation(
        segmentation_dir=str(tmp_path),
        mask_files=filenames,
        smoothing=0.0,
        brain_face_count=None,
        scalp_face_count=None,
    )
    point_type = cedalion.dataclasses.PointType
    geo3d = cedalion.dataclasses.build_labeled_points(
        [[8.5, 8.5, 16.0], [8.5, 14.5, 13.0]],
        crs="ijk",
        units="1",
        labels=["S1", "D1"],
        types=[point_type.SOURCE, point_type.DETECTOR],
    )
    measurement_list = pd.DataFrame(
        {
            "channel": ["S1D1"],
            "source": ["S1"],
            "detector": ["D1"],
            "wavelength": [760.0],
        }
    )

    result = core.run_forward_model(
        head,
        geo3d,
        measurement_list,
        solver="mcx_opencl",
        output_dir=tmp_path / "forward",
        nphoton=1_000,
    )

    assert result.adot.sizes["channel"] == 1
    assert result.adot.sizes["wavelength"] == 1
    assert result.adot.sizes["vertex"] > 0
    assert result.fluence_path.exists()
    assert result.adot_path.exists()
    assert np.isfinite(result.adot).all()
    assert float(np.max(result.adot)) > 0
    assert result.parameters["solver"] == "mcx_opencl"
