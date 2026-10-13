"""Align digitized fNIRS optodes to a Cedalion standard head and visualize labels.

The input JSON is an extraction of the user-supplied workbook with records containing
``label``, ``x``, ``y`` and ``z``.  Landmark aliases are normalized as follows:
NzHS -> Nz, IzHS -> Iz, ARHS -> RPA, ALHS -> LPA, CzHS -> Cz.

This script intentionally performs geometry/visualization only.  It does not infer a
measurement list and it does not perform DOT reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import cKDTree
import trimesh


LANDMARK_ALIASES = {
    "NzHS": "Nz",
    "IzHS": "Iz",
    "ARHS": "RPA",
    "ALHS": "LPA",
    "CzHS": "Cz",
}


def load_points(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["label"]: np.asarray([row["x"], row["y"], row["z"]], dtype=float)
        for row in payload["rows"]
    }


def load_slicer_landmarks(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, np.ndarray] = {}
    for markup in payload["markups"]:
        for point in markup["controlPoints"]:
            result[point["label"]] = np.asarray(point["position"], dtype=float)
    return result


def similarity_transform(source: np.ndarray, target: np.ndarray):
    """Return a proper-rotation similarity transform from source to target."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    x = source - source_mean
    y = target - target_mean
    covariance = y.T @ x / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    source_variance = np.sum(x**2) / len(source)
    scale = np.sum(singular_values * np.diag(correction)) / source_variance
    translation = target_mean - scale * (rotation @ source_mean)
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return matrix, scale, rotation


def affine_transform(source: np.ndarray, target: np.ndarray):
    """Return the least-squares 12-parameter affine transform."""
    design = np.column_stack([source, np.ones(len(source))])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    matrix = np.eye(4)
    matrix[:3, :] = coefficients.T
    return matrix


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (homogeneous @ matrix.T)[:, :3]


def residuals(source: np.ndarray, target: np.ndarray, matrix: np.ndarray):
    fitted = apply_transform(source, matrix)
    errors = np.linalg.norm(fitted - target, axis=1)
    return fitted, errors


def set_equal_axes(ax, vertices: np.ndarray):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2
    radius = np.max(maxs - mins) / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coords-json", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("similarity", "affine"), default="similarity"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    measured = load_points(args.coords_json)
    atlas = load_slicer_landmarks(args.head_dir / "landmarks.mrk.json")
    anchor_labels = ["Nz", "Iz", "LPA", "RPA", "Cz"]
    measured_anchors = np.vstack(
        [measured[next(k for k, v in LANDMARK_ALIASES.items() if v == label)]
         for label in anchor_labels]
    )
    atlas_anchors = np.vstack([atlas[label] for label in anchor_labels])

    similarity, scale, rotation = similarity_transform(measured_anchors, atlas_anchors)
    affine = affine_transform(measured_anchors, atlas_anchors)
    _, similarity_errors = residuals(measured_anchors, atlas_anchors, similarity)
    _, affine_errors = residuals(measured_anchors, atlas_anchors, affine)
    matrix = similarity if args.method == "similarity" else affine

    labels = list(measured)
    raw_points = np.vstack([measured[label] for label in labels])
    aligned_points = apply_transform(raw_points, matrix)

    scalp = trimesh.load(args.head_dir / "mask_scalp.obj", process=False)
    scalp_vertices_ijk = np.asarray(scalp.vertices)
    scalp_faces = np.asarray(scalp.faces)
    ijk_to_ras = nib.load(args.head_dir / "mask_skin.nii").affine
    scalp_vertices = apply_transform(scalp_vertices_ijk, ijk_to_ras)
    scalp_tree = cKDTree(scalp_vertices)
    surface_distance, nearest_vertex = scalp_tree.query(aligned_points)
    snapped_points = scalp_vertices[nearest_vertex]

    landmark_set = set(LANDMARK_ALIASES)
    optode_mask = np.asarray([label not in landmark_set for label in labels])
    source_mask = np.asarray([label.startswith("S") for label in labels])
    detector_mask = np.asarray([label.startswith("D") for label in labels])

    standard_labels = [label for label in atlas if label not in anchor_labels]
    standard_points = np.vstack([atlas[label] for label in standard_labels])
    standard_tree = cKDTree(standard_points)
    label_distance, label_index = standard_tree.query(snapped_points)

    csv_path = args.output_dir / "optode_to_1010_labels.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "optode", "type", "aligned_x_mm", "aligned_y_mm", "aligned_z_mm",
            "snapped_x_mm", "snapped_y_mm", "snapped_z_mm",
            "distance_to_scalp_mm", "nearest_1010_label",
            "distance_to_1010_label_mm",
        ])
        for i, label in enumerate(labels):
            if not optode_mask[i]:
                continue
            kind = "source" if source_mask[i] else "detector"
            writer.writerow([
                label, kind, *aligned_points[i], *snapped_points[i],
                surface_distance[i], standard_labels[label_index[i]], label_distance[i],
            ])

    qc = {
        "input_sheet": "p4_others",
        "input_coordinate_unit_inference": "centimeter-like; scale estimated by registration",
        "anchor_mapping": LANDMARK_ALIASES,
        "selected_method": args.method,
        "similarity_scale_to_mm": float(scale),
        "similarity_rotation_determinant": float(np.linalg.det(rotation)),
        "similarity_anchor_errors_mm": dict(zip(anchor_labels, similarity_errors.tolist())),
        "similarity_anchor_rmse_mm": float(np.sqrt(np.mean(similarity_errors**2))),
        "affine_anchor_errors_mm": dict(zip(anchor_labels, affine_errors.tolist())),
        "affine_anchor_rmse_mm": float(np.sqrt(np.mean(affine_errors**2))),
        "affine_linear_determinant": float(np.linalg.det(affine[:3, :3])),
        "optode_count": int(optode_mask.sum()),
        "source_count": int(source_mask.sum()),
        "detector_count": int(detector_mask.sum()),
        "median_pre_snap_distance_mm": float(np.median(surface_distance[optode_mask])),
        "max_pre_snap_distance_mm": float(np.max(surface_distance[optode_mask])),
        "transform_matrix": matrix.tolist(),
        "head_surface_crs": "Colin27 MNI/RAS millimeters",
    }
    (args.output_dir / "registration_qc.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    figure = plt.figure(figsize=(16, 11), constrained_layout=True)
    views = [(25, -80, "Left-oblique"), (25, 10, "Front"), (90, -90, "Top")]
    sampled_faces = scalp_faces[::6]
    for panel, (elev, azim, title) in enumerate(views, start=1):
        ax = figure.add_subplot(2, 2, panel, projection="3d")
        ax.plot_trisurf(
            scalp_vertices[:, 0], scalp_vertices[:, 1], scalp_vertices[:, 2],
            triangles=sampled_faces, color="#D7C7B8", alpha=0.14,
            linewidth=0.1, edgecolor="#A89787",
        )
        ax.scatter(
            snapped_points[source_mask, 0], snapped_points[source_mask, 1],
            snapped_points[source_mask, 2], c="#D62728", s=20, label="Sources",
            depthshade=False,
        )
        ax.scatter(
            snapped_points[detector_mask, 0], snapped_points[detector_mask, 1],
            snapped_points[detector_mask, 2], c="#1F77B4", s=17,
            label="Detectors", depthshade=False,
        )
        ax.scatter(
            atlas_anchors[:, 0], atlas_anchors[:, 1], atlas_anchors[:, 2],
            c="#2CA02C", s=38, marker="^", label="Registration landmarks",
            depthshade=False,
        )
        display_labels = {
            "Nz", "Iz", "LPA", "RPA", "Cz", "Fpz", "Fz", "FCz", "C3", "C4",
            "Cz", "CPz", "Pz", "Oz", "F7", "F8", "T7", "T8", "P7", "P8",
        }
        for label in display_labels:
            point = atlas[label]
            ax.text(point[0], point[1], point[2], label, fontsize=5.5, color="#222222")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel("R-L / X (mm)")
        ax.set_ylabel("A-P / Y (mm)")
        ax.set_zlabel("I-S / Z (mm)")
        set_equal_axes(ax, scalp_vertices)
        if panel == 1:
            ax.legend(loc="upper left", fontsize=8)

    ax = figure.add_subplot(2, 2, 4, projection="3d")
    ax.scatter(
        raw_points[source_mask, 0], raw_points[source_mask, 1],
        raw_points[source_mask, 2], c="#D62728", s=20, label="Sources",
    )
    ax.scatter(
        raw_points[detector_mask, 0], raw_points[detector_mask, 1],
        raw_points[detector_mask, 2], c="#1F77B4", s=17, label="Detectors",
    )
    for original, canonical in LANDMARK_ALIASES.items():
        p = measured[original]
        ax.scatter(*p, c="#2CA02C", s=40, marker="^")
        ax.text(*p, canonical, fontsize=8)
    ax.set_title("Original measured coordinate frame")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_equal_axes(ax, raw_points)
    ax.legend(loc="upper left", fontsize=8)

    figure.suptitle(
        f"fNIRS optodes registered to Colin27 10-10 landmarks ({args.method})\n"
        f"Similarity anchor RMSE={qc['similarity_anchor_rmse_mm']:.1f} mm; "
        f"Affine RMSE={qc['affine_anchor_rmse_mm']:.1f} mm",
        fontsize=14,
    )
    figure.savefig(args.output_dir / "registered_optodes_and_1010_labels.png", dpi=220)
    plt.close(figure)

    optode_indices = np.flatnonzero(optode_mask)
    optode_hover = [
        (
            f"{labels[i]}<br>nearest 10-10: {standard_labels[label_index[i]]}"
            f"<br>label distance: {label_distance[i]:.1f} mm"
            f"<br>pre-snap scalp distance: {surface_distance[i]:.1f} mm"
        )
        for i in optode_indices
    ]
    plot_faces = scalp_faces[::4]
    interactive = go.Figure()
    interactive.add_trace(go.Mesh3d(
        x=scalp_vertices[:, 0], y=scalp_vertices[:, 1], z=scalp_vertices[:, 2],
        i=plot_faces[:, 0], j=plot_faces[:, 1], k=plot_faces[:, 2],
        color="#D7C7B8", opacity=0.16, name="Colin27 scalp", hoverinfo="skip",
    ))
    interactive.add_trace(go.Scatter3d(
        x=snapped_points[source_mask, 0], y=snapped_points[source_mask, 1],
        z=snapped_points[source_mask, 2], mode="markers", name="Sources",
        marker={"size": 5, "color": "#D62728"},
        text=[optode_hover[list(optode_indices).index(i)] for i in np.flatnonzero(source_mask)],
        hovertemplate="%{text}<extra></extra>",
    ))
    interactive.add_trace(go.Scatter3d(
        x=snapped_points[detector_mask, 0], y=snapped_points[detector_mask, 1],
        z=snapped_points[detector_mask, 2], mode="markers", name="Detectors",
        marker={"size": 4.5, "color": "#1F77B4"},
        text=[optode_hover[list(optode_indices).index(i)] for i in np.flatnonzero(detector_mask)],
        hovertemplate="%{text}<extra></extra>",
    ))
    interactive.add_trace(go.Scatter3d(
        x=standard_points[:, 0], y=standard_points[:, 1], z=standard_points[:, 2],
        mode="markers", name="10-10 labels", text=standard_labels,
        marker={"size": 3.5, "color": "#222222", "symbol": "diamond"},
        hovertemplate="10-10 label: %{text}<extra></extra>",
    ))
    interactive.add_trace(go.Scatter3d(
        x=atlas_anchors[:, 0], y=atlas_anchors[:, 1], z=atlas_anchors[:, 2],
        mode="markers+text", name="Registration landmarks", text=anchor_labels,
        textposition="top center", marker={"size": 7, "color": "#2CA02C", "symbol": "diamond"},
        hovertemplate="anchor: %{text}<extra></extra>",
    ))
    interactive.update_layout(
        title=(
            f"fNIRS optodes and Colin27 10-10 labels ({args.method}); "
            f"anchor RMSE {qc[args.method + '_anchor_rmse_mm']:.1f} mm"
        ),
        scene={
            "aspectmode": "data",
            "xaxis_title": "R-L / X (mm)",
            "yaxis_title": "A-P / Y (mm)",
            "zaxis_title": "I-S / Z (mm)",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "b": 0, "t": 55},
    )
    interactive.write_html(
        args.output_dir / "registered_optodes_interactive.html",
        include_plotlyjs=True,
    )

    print(json.dumps(qc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
