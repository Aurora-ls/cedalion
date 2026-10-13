"""Streamlit front end for Cedalion DOT registration and reconstruction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import core  # noqa: E402


st.set_page_config(
    page_title="Cedalion DOT Web",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


STATE_DEFAULTS = {
    "workspace": None,
    "geo3d": None,
    "measurement_list": None,
    "recording": None,
    "data_keys": [],
    "head_model": None,
    "head_source": None,
    "coordinate_source": None,
    "measurement_source": None,
    "registration_result": None,
    "forward_result": None,
    "adot": None,
    "reconstruction": None,
}


def initialize_state() -> None:
    for key, value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.workspace is None:
        st.session_state.workspace = core.make_session_workspace()


def reset_downstream(stage: str) -> None:
    order = ["registration_result", "forward_result", "adot", "reconstruction"]
    start = order.index(stage)
    for key in order[start:]:
        st.session_state[key] = None


def status_label(value, yes: str, no: str) -> str:
    return f"✅ {yes}" if value is not None else f"○ {no}"


def point_values_mm(points) -> np.ndarray:
    return points.pint.to("mm").pint.dequantify().values


def _mesh_trace(surface, name: str, color: str, opacity: float, intensity=None):
    vertices = np.asarray(surface.mesh.vertices)
    faces = np.asarray(surface.mesh.faces)
    max_faces = 45_000
    if len(faces) > max_faces:
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=int)
        faces = faces[indices]
    kwargs = {
        "x": vertices[:, 0],
        "y": vertices[:, 1],
        "z": vertices[:, 2],
        "i": faces[:, 0],
        "j": faces[:, 1],
        "k": faces[:, 2],
        "name": name,
        "opacity": opacity,
        "flatshading": False,
        "hoverinfo": "skip",
    }
    if intensity is None:
        kwargs["color"] = color
    else:
        kwargs.update(
            intensity=np.asarray(intensity, dtype=float),
            colorscale="RdBu_r",
            colorbar={"title": "重建值"},
            showscale=True,
        )
    return go.Mesh3d(**kwargs)


def reconstruction_brain_values(image, head_ras):
    if image is None or "vertex" not in image.dims:
        return None
    selected = image
    non_vertex_dims = [dim for dim in selected.dims if dim != "vertex"]
    for dim in non_vertex_dims:
        selected = selected.isel({dim: st.session_state.get(f"viewer_{dim}", 0)})
    values = np.asarray(selected.pint.dequantify()).squeeze()
    if values.ndim != 1:
        return None
    if len(values) == head_ras.brain.nvertices:
        return values
    if "is_brain" in selected.coords:
        mask = np.asarray(selected.is_brain.values, dtype=bool)
        values = values[mask]
        if len(values) == head_ras.brain.nvertices:
            return values
    return None


def build_scene(
    *,
    show_scalp: bool,
    show_brain: bool,
    show_optodes: bool,
    show_landmarks: bool,
    show_channels: bool,
    show_reconstruction: bool,
) -> go.Figure:
    figure = go.Figure()
    registration = st.session_state.registration_result
    head = registration.head_ras if registration is not None else None
    if head is None and st.session_state.head_model is not None:
        head = core.head_to_ras(st.session_state.head_model)

    geo = (
        registration.registered_geo3d
        if registration is not None
        else st.session_state.geo3d
    )
    reconstruction_values = None
    if head is not None and show_reconstruction:
        reconstruction_values = reconstruction_brain_values(
            st.session_state.reconstruction, head
        )

    if head is not None and show_scalp:
        figure.add_trace(_mesh_trace(head.scalp, "头皮", "#D8B4A0", 0.18))
    if head is not None and show_brain:
        figure.add_trace(
            _mesh_trace(
                head.brain,
                "脑表面",
                "#D7DCE2",
                0.80 if reconstruction_values is not None else 0.34,
                reconstruction_values,
            )
        )

    if geo is not None:
        points = point_values_mm(geo)
        labels = geo.label.values.astype(str)
        type_names = np.array(
            [getattr(value, "name", str(value)) for value in geo.type.values]
        )
        marker_groups = [
            ("SOURCE", "光源", "#E53935", 6),
            ("DETECTOR", "探测器", "#1976D2", 6),
        ]
        if show_optodes:
            for type_name, name, color, size in marker_groups:
                mask = type_names == type_name
                if np.any(mask):
                    figure.add_trace(
                        go.Scatter3d(
                            x=points[mask, 0],
                            y=points[mask, 1],
                            z=points[mask, 2],
                            mode="markers",
                            name=name,
                            text=labels[mask],
                            marker={"size": size, "color": color},
                            hovertemplate=(
                                "%{text}<br>(%{x:.1f}, %{y:.1f}, %{z:.1f}) mm"
                                "<extra></extra>"
                            ),
                        )
                    )
        if show_landmarks:
            mask = type_names == "LANDMARK"
            if np.any(mask):
                figure.add_trace(
                    go.Scatter3d(
                        x=points[mask, 0],
                        y=points[mask, 1],
                        z=points[mask, 2],
                        mode="markers+text",
                        name="解剖标志点",
                        text=labels[mask],
                        textposition="top center",
                        marker={"size": 8, "color": "#2E7D32", "symbol": "diamond"},
                        hovertemplate="%{text}<extra></extra>",
                    )
                )
        if show_channels and st.session_state.measurement_list is not None:
            by_label = {label: points[index] for index, label in enumerate(labels)}
            unique_pairs = st.session_state.measurement_list[
                ["channel", "source", "detector"]
            ].drop_duplicates()
            xs, ys, zs = [], [], []
            for row in unique_pairs.itertuples(index=False):
                if row.source not in by_label or row.detector not in by_label:
                    continue
                source = by_label[row.source]
                detector = by_label[row.detector]
                xs.extend([source[0], detector[0], None])
                ys.extend([source[1], detector[1], None])
                zs.extend([source[2], detector[2], None])
            figure.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    name="测量通道",
                    line={"color": "rgba(70,70,70,0.45)", "width": 2},
                    hoverinfo="skip",
                )
            )

    figure.update_layout(
        height=720,
        margin={"l": 0, "r": 0, "t": 35, "b": 0},
        legend={"orientation": "h", "y": 1.02},
        scene={
            "aspectmode": "data",
            "xaxis_title": "R-L / X (mm)",
            "yaxis_title": "A-P / Y (mm)",
            "zaxis_title": "I-S / Z (mm)",
        },
    )
    return figure


def render_summary_cards() -> None:
    columns = st.columns(5)
    values = [
        status_label(st.session_state.geo3d, "坐标已导入", "等待坐标"),
        status_label(st.session_state.measurement_list, "通道已导入", "等待通道"),
        status_label(st.session_state.head_model, "头模已导入", "等待头模"),
        status_label(st.session_state.registration_result, "配准完成", "等待配准"),
        status_label(st.session_state.adot, "Adot可用", "等待Adot"),
    ]
    for column, value in zip(columns, values, strict=True):
        column.markdown(f"**{value}**")


initialize_state()

st.title("Cedalion DOT 3D 重建 Web 工作台")
st.caption(
    "本地运行的研究型 MVP：坐标/通道/SNIRF 导入 → 头模 → 一键配准 → "
    "MCX/NIRFASTer 正向模型 → Adot → 正则化 DOT 逆重建。"
)
st.warning(
    "当前 Cedalion 正向模型中的多波长 fluence/组织参数路径仍有 FIXME。"
    "软件可以执行并保存结果，"
    "但在完成波长相关验证前，不应将输出表述为已经定量验证的临床 DOT。",
    icon="⚠️",
)
render_summary_cards()


with st.sidebar:
    st.header("三维显示")
    show_scalp = st.checkbox("头皮模型", value=True)
    show_brain = st.checkbox("脑表面", value=True)
    show_optodes = st.checkbox("光源/探测器", value=True)
    show_landmarks = st.checkbox("解剖标志点", value=True)
    show_channels = st.checkbox("测量通道连线", value=False)
    show_reconstruction = st.checkbox("重建地形图", value=True)
    st.divider()
    st.header("求解器状态")
    capabilities = core.solver_capabilities(REPO_ROOT)
    for info in capabilities.values():
        icon = "✅" if info["available"] else "⛔"
        st.caption(f"{icon} {info['label']}")
        st.caption(info["detail"])
    st.divider()
    st.caption(f"会话目录：`{st.session_state.workspace}`")


tabs = st.tabs(
    [
        "1 数据导入",
        "2 头模型",
        "3 自动配准",
        "4 Adot 正向模型",
        "5 DOT 重建",
        "6 三维查看与导出",
    ]
)


with tabs[0]:
    st.subheader("SNIRF 一体化导入")
    snirf_file = st.file_uploader(
        "导入 SNIRF（可同时提供光极、解剖点、通道定义和测量数据）",
        type=["snirf"],
        key="snirf_upload",
    )
    if st.button("读取 SNIRF", disabled=snirf_file is None, use_container_width=True):
        try:
            path = core.save_uploaded_file(
                snirf_file.name,
                snirf_file.getvalue(),
                st.session_state.workspace / "inputs",
            )
            with st.spinner("正在读取 SNIRF…"):
                bundle = core.read_snirf(path)
            st.session_state.recording = bundle.recording
            st.session_state.geo3d = bundle.geo3d
            st.session_state.measurement_list = bundle.measurement_list
            st.session_state.data_keys = bundle.data_keys
            st.session_state.coordinate_source = snirf_file.name
            st.session_state.measurement_source = snirf_file.name
            reset_downstream("registration_result")
            st.success(
                f"已读取 {len(bundle.geo3d.label)} 个点、"
                f"{bundle.measurement_list.channel.nunique()} 个通道。"
            )
            for warning in bundle.warnings:
                st.warning(warning)
        except Exception as exc:
            st.exception(exc)

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("单独导入坐标")
        coordinate_file = st.file_uploader(
            "坐标表：CSV / TSV / JSON / XLSX",
            type=["csv", "tsv", "txt", "json", "xlsx", "xlsm"],
            key="coordinate_upload",
        )
        unit = st.selectbox("坐标单位", ["mm", "cm", "m"], index=0)
        crs = st.text_input("坐标系名称", value="digitized")
        if st.button(
            "读取坐标表", disabled=coordinate_file is None, use_container_width=True
        ):
            try:
                frame = core.read_table_bytes(
                    coordinate_file.name, coordinate_file.getvalue()
                )
                geo3d = core.build_labeled_points(frame, units=unit, crs=crs)
                st.session_state.geo3d = geo3d
                st.session_state.coordinate_source = coordinate_file.name
                reset_downstream("registration_result")
                st.success(f"已读取 {len(geo3d.label)} 个坐标点。")
            except Exception as exc:
                st.exception(exc)

    with right:
        st.subheader("单独导入测量通道")
        measurement_file = st.file_uploader(
            "通道表：channel / source / detector / wavelength",
            type=["csv", "tsv", "txt", "json", "xlsx", "xlsm"],
            key="measurement_upload",
        )
        if st.button(
            "读取通道表", disabled=measurement_file is None, use_container_width=True
        ):
            try:
                frame = core.read_table_bytes(
                    measurement_file.name, measurement_file.getvalue()
                )
                labels = (
                    st.session_state.geo3d.label.values
                    if st.session_state.geo3d is not None
                    else None
                )
                measurement_list = core.normalize_measurement_list(frame, labels)
                st.session_state.measurement_list = measurement_list
                st.session_state.measurement_source = measurement_file.name
                reset_downstream("registration_result")
                st.success(
                    f"已读取 {measurement_list.channel.nunique()} 个真实通道、"
                    f"{measurement_list.wavelength.nunique()} 个波长。"
                )
            except Exception as exc:
                st.exception(exc)

    if st.session_state.geo3d is not None:
        st.markdown("**坐标预览**")
        st.dataframe(
            core._points_to_frame(st.session_state.geo3d), use_container_width=True
        )
    if st.session_state.measurement_list is not None:
        st.markdown("**测量列表预览**")
        st.dataframe(st.session_state.measurement_list, use_container_width=True)


with tabs[1]:
    st.subheader("选择头模型")
    source_type = st.radio(
        "头模来源",
        ["Colin27 标准头模", "ICBM152 标准头模", "自定义头模 ZIP"],
        horizontal=True,
    )
    custom_head_file = None
    if source_type == "自定义头模 ZIP":
        custom_head_file = st.file_uploader(
            "上传 Cedalion save() 目录 ZIP，或分割掩膜/表面 ZIP",
            type=["zip"],
            key="head_upload",
        )
        st.caption(
            "推荐 ZIP 内包含 Cedalion `TwoSurfaceHeadModel.save()` 的完整文件；"
            "也支持默认命名的 mask_csf/mask_gray/mask_skin/mask_bone/"
            "mask_white NIfTI 与 landmarks.mrk.json。"
        )
    disabled = source_type == "自定义头模 ZIP" and custom_head_file is None
    if st.button(
        "加载头模型", disabled=disabled, type="primary", use_container_width=True
    ):
        try:
            with st.spinner("正在加载/构建头模型，首次加载标准模型可能需要下载…"):
                if source_type.startswith("Colin27"):
                    head_model = core.load_standard_head_model("colin27")
                elif source_type.startswith("ICBM152"):
                    head_model = core.load_standard_head_model("icbm152")
                else:
                    head_model = core.load_custom_head_model_zip(
                        custom_head_file.getvalue(), st.session_state.workspace
                    )
            st.session_state.head_model = head_model
            st.session_state.head_source = source_type
            reset_downstream("registration_result")
            st.success("头模型加载完成。")
        except Exception as exc:
            st.exception(exc)
    if st.session_state.head_model is not None:
        st.code(str(st.session_state.head_model))


with tabs[2]:
    st.subheader("一键头模型配准")
    mode_label = st.selectbox(
        "全局配准方式",
        [
            "相似变换：平移+旋转+等比缩放（推荐）",
            "刚体：平移+旋转",
            "全仿射：允许非等比缩放/剪切",
            "坐标已同系：仅单位转换",
        ],
    )
    mode_map = {
        "相似变换：平移+旋转+等比缩放（推荐）": "trans_rot_isoscale",
        "刚体：平移+旋转": "trans_rot",
        "全仿射：允许非等比缩放/剪切": "general",
        "坐标已同系：仅单位转换": "identity",
    }
    use_spring = st.checkbox("使用通道约束的弹簧 ICP 吸附到头皮", value=True)
    with st.expander("高级参数"):
        n_iter = st.number_input("最大迭代次数", 10, 2000, 400, 10)
        k_spring = st.number_input("通道弹簧系数", 0.01, 100.0, 1.0, 0.1)
        k_anchor = st.number_input("解剖点锚定系数", 0.01, 100.0, 10.0, 0.5)
        step_size = st.number_input("迭代步长", 0.001, 1.0, 0.1, 0.01)
        convergence_tol = st.number_input(
            "收敛阈值（头模单位）", 0.0001, 10.0, 0.01, 0.01, format="%.4f"
        )

    registration_ready = (
        st.session_state.geo3d is not None
        and st.session_state.head_model is not None
        and (not use_spring or st.session_state.measurement_list is not None)
    )
    if st.button(
        "一键配准",
        disabled=not registration_ready,
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("正在执行解剖点全局配准和头皮吸附…"):
                result = core.register_geometry(
                    st.session_state.head_model,
                    st.session_state.geo3d,
                    st.session_state.measurement_list,
                    mode=mode_map[mode_label],
                    use_spring_icp=use_spring,
                    n_iter=n_iter,
                    k_spring=k_spring,
                    k_anchor=k_anchor,
                    step_size=step_size,
                    convergence_tol=convergence_tol,
                )
            st.session_state.registration_result = result
            reset_downstream("forward_result")
            st.success("配准完成。")
        except Exception as exc:
            st.exception(exc)

    result = st.session_state.registration_result
    if result is not None:
        metrics = st.columns(3)
        metrics[0].metric("共同解剖点", len(result.parameters["common_landmarks"]))
        metrics[1].metric(
            "初始 RMSE",
            f"{result.parameters['initial_landmark_rmse_mm']:.2f} mm",
        )
        metrics[2].metric(
            "最终 RMSE",
            f"{result.parameters['final_landmark_rmse_mm']:.2f} mm",
        )
        st.json(result.parameters, expanded=False)
        st.download_button(
            "下载配准参数 JSON",
            core.json_bytes(result.parameters),
            "registration_parameters.json",
            "application/json",
            use_container_width=True,
        )
        st.download_button(
            "下载配准后坐标 CSV",
            result.registered_coordinates.to_csv(index=False).encode("utf-8-sig"),
            "registered_coordinates.csv",
            "text/csv",
            use_container_width=True,
        )


with tabs[3]:
    st.subheader("正向光传播与 Adot 灵敏度矩阵")
    solver_options = list(capabilities)
    solver = st.selectbox(
        "求解器",
        solver_options,
        format_func=lambda value: capabilities[value]["label"],
    )
    solver_info = capabilities[solver]
    if solver_info["available"]:
        st.success(f"求解器可用：{solver_info['requirement']}；{solver_info['detail']}")
    else:
        st.error(
            f"当前环境不可用：需要 {solver_info['requirement']}；"
            f"检测结果：{solver_info['detail']}"
        )
    nphoton = st.number_input(
        "MCX 光子数",
        min_value=100_000,
        max_value=1_000_000_000,
        value=10_000_000,
        step=1_000_000,
        disabled=solver == "nirfaster",
    )
    st.caption(
        "只使用导入 measurement list 中的真实 source-detector 组合；"
        "不会生成全部光源×探测器组合。"
    )
    forward_ready = (
        st.session_state.registration_result is not None
        and st.session_state.measurement_list is not None
        and solver_info["available"]
    )
    if st.button(
        "运行正向模型并计算 Adot",
        disabled=not forward_ready,
        type="primary",
        use_container_width=True,
    ):
        try:
            output_dir = st.session_state.workspace / "forward"
            with st.spinner("正在进行光传播模拟；高光子数/高分辨率头模可能耗时很长…"):
                forward = core.run_forward_model(
                    st.session_state.registration_result.head_ras,
                    st.session_state.registration_result.registered_geo3d,
                    st.session_state.measurement_list,
                    solver=solver,
                    output_dir=output_dir,
                    nphoton=nphoton,
                )
            st.session_state.forward_result = forward
            st.session_state.adot = forward.adot
            st.session_state.reconstruction = None
            st.success("Adot 计算完成。")
        except Exception as exc:
            st.exception(exc)

    st.divider()
    uploaded_adot = st.file_uploader(
        "或者导入已有 Adot NetCDF",
        type=["nc", "netcdf"],
        key="adot_upload",
    )
    if st.button(
        "读取已有 Adot", disabled=uploaded_adot is None, use_container_width=True
    ):
        try:
            st.session_state.adot = core.load_adot_bytes(
                uploaded_adot.getvalue(), st.session_state.workspace / "inputs"
            )
            st.session_state.forward_result = None
            st.session_state.reconstruction = None
            st.success("Adot 已载入。")
        except Exception as exc:
            st.exception(exc)

    if st.session_state.adot is not None:
        adot = st.session_state.adot
        st.write("Adot 维度：", dict(adot.sizes))
        st.download_button(
            "下载 Adot.nc",
            core.dataarray_to_netcdf_bytes(adot),
            "Adot.nc",
            "application/x-netcdf",
            use_container_width=True,
        )
        st.download_button(
            "下载 Adot.npz",
            core.dataarray_to_npz_bytes(adot),
            "Adot.npz",
            "application/octet-stream",
            use_container_width=True,
        )
        if st.session_state.forward_result is not None:
            st.download_button(
                "下载正向模型参数 JSON",
                core.json_bytes(st.session_state.forward_result.parameters),
                "forward_model_parameters.json",
                "application/json",
                use_container_width=True,
            )
            st.caption(
                f"fluence 保存在：`{st.session_state.forward_result.fluence_path}`"
            )


with tabs[4]:
    st.subheader("正则化 DOT 逆重建")
    if st.session_state.recording is None:
        st.info("请先导入包含测量数据的 SNIRF。")
    data_keys = st.session_state.data_keys or []
    default_key = "od" if "od" in data_keys else ("amp" if "amp" in data_keys else None)
    selected_key = (
        st.selectbox(
            "测量数据",
            data_keys,
            index=data_keys.index(default_key) if default_key in data_keys else 0,
            disabled=not data_keys,
        )
        if data_keys
        else None
    )
    recon_mode = st.selectbox(
        "重建模式",
        ["mua2conc", "conc", "mua"],
        format_func=lambda value: {
            "mua2conc": "先重建吸收变化，再转换 HbO/HbR",
            "conc": "直接重建 HbO/HbR 浓度",
            "mua": "逐波长重建吸收系数",
        }[value],
    )
    brain_only = st.checkbox("仅重建脑表面顶点", value=True)
    alpha_meas = st.number_input("测量正则化 α", 1e-8, 10.0, 0.01, format="%.6f")
    use_spatial = st.checkbox("启用空间/深度正则化", value=False)
    alpha_spatial = st.number_input(
        "空间正则化 α",
        1e-8,
        100.0,
        0.1,
        format="%.6f",
        disabled=not use_spatial,
    )
    reconstruction_ready = (
        st.session_state.adot is not None
        and st.session_state.recording is not None
        and selected_key is not None
    )
    if st.button(
        "运行 3D DOT 重建",
        disabled=not reconstruction_ready,
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("正在构建正则化逆矩阵并重建…"):
                od = core.prepare_optical_density(
                    st.session_state.recording, selected_key
                )
                image = core.run_reconstruction(
                    st.session_state.adot,
                    od,
                    recon_mode=recon_mode,
                    brain_only=brain_only,
                    alpha_meas=alpha_meas,
                    alpha_spatial=alpha_spatial if use_spatial else None,
                )
            st.session_state.reconstruction = image
            st.success("DOT 重建完成。")
        except Exception as exc:
            st.exception(exc)

    if st.session_state.reconstruction is not None:
        image = st.session_state.reconstruction
        st.write("重建结果维度：", dict(image.sizes))
        st.download_button(
            "下载 reconstruction.nc",
            core.dataarray_to_netcdf_bytes(image),
            "reconstruction.nc",
            "application/x-netcdf",
            use_container_width=True,
        )


with tabs[5]:
    st.subheader("三维场景")
    image = st.session_state.reconstruction
    if image is not None:
        selectable_dims = [dim for dim in image.dims if dim != "vertex"]
        if selectable_dims:
            controls = st.columns(min(len(selectable_dims), 4))
            for index, dim in enumerate(selectable_dims):
                size = image.sizes[dim]
                coord = image.coords.get(dim)
                options = list(range(size))
                selected = controls[index % len(controls)].select_slider(
                    f"显示 {dim}",
                    options=options,
                    value=min(st.session_state.get(f"viewer_{dim}", 0), size - 1),
                    format_func=(
                        (lambda i, coord=coord: str(coord.values[i]))
                        if coord is not None
                        else str
                    ),
                    key=f"viewer_{dim}",
                )
    if st.session_state.geo3d is None and st.session_state.head_model is None:
        st.info("导入坐标或头模型后显示三维场景。")
    else:
        st.plotly_chart(
            build_scene(
                show_scalp=show_scalp,
                show_brain=show_brain,
                show_optodes=show_optodes,
                show_landmarks=show_landmarks,
                show_channels=show_channels,
                show_reconstruction=show_reconstruction,
            ),
            use_container_width=True,
        )

    manifest = core.make_project_manifest(
        coordinate_source=st.session_state.coordinate_source,
        measurement_source=st.session_state.measurement_source,
        head_model_source=st.session_state.head_source,
        registration=(
            st.session_state.registration_result.parameters
            if st.session_state.registration_result is not None
            else None
        ),
        forward=st.session_state.forward_result,
        reconstruction=st.session_state.reconstruction,
    )
    st.download_button(
        "下载项目清单 JSON",
        core.json_bytes(manifest),
        "dot_project_manifest.json",
        "application/json",
        use_container_width=True,
    )
