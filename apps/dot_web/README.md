# Cedalion DOT Web

一个基于本地 Cedalion checkout 的 Streamlit 研究型 Web MVP。它把以下流程放在同一个浏览器工作台中：

1. 导入 SNIRF，或分别导入光极/解剖点坐标与 measurement list。
2. 加载 Colin27、ICBM152 或自定义 `TwoSurfaceHeadModel`。
3. 通过解剖标志点完成全局配准，并可使用通道约束的弹簧 ICP 吸附到头皮。
4. 调用 Cedalion `ForwardModel` 运行 MCX/MCXCL/NIRFASTer，计算并保存 Adot。
5. 对 SNIRF 中的 OD 或原始强度数据调用 `ImageRecon`，显示脑表面 3D 重建结果。

## 启动

优先使用本机 `py312` 环境：

```powershell


conda run -n py312 python -m pip install -e .
conda run -n py312 python -m pip install -r .\apps\dot_web\requirements.txt
.\apps\dot_web\run.ps1
```

浏览器默认打开 `http://localhost:8501`。

## 输入格式

### 坐标表

支持 CSV、TSV、JSON、XLSX，至少包含：

```text
label,x,y,z,type
Nz,0,90,-35,landmark
S01,-30,45,70,source
D01,-20,50,68,detector
```

`type` 可省略；以 `S`/`D` 开头的标签会自动识别。`NzHS/IzHS/ARHS/ALHS/CzHS` 会映射为 `Nz/Iz/RPA/LPA/Cz`。

### Measurement list

一行对应一个“通道—波长”测量：

```text
channel,source,detector,wavelength
ch001,S01,D01,760
ch001,S01,D01,850
```

应用只使用表中真实通道，不推断全部 source × detector 组合。

### 自定义头模 ZIP

推荐将 `TwoSurfaceHeadModel.save(folder)` 生成的目录整体压缩。应用也支持包含以下默认文件的分割包：

- `mask_csf.nii`
- `mask_gray.nii`
- `mask_skin.nii`
- `mask_bone.nii`
- `mask_white.nii`
- 可选 `landmarks.mrk.json`
- 可选 `mask_brain.obj` 与 `mask_scalp.obj`

非默认文件名可通过 ZIP 内的 `head_model.json` 指定 `mask_files`、`landmarks_ras_file`、`brain_surface_file` 和 `scalp_surface_file`。

## 输出

- `registration_parameters.json`：全局变换矩阵和配准 QC。弹簧 ICP 属于非刚性调整，因此最终坐标另存为 CSV。
- `registered_coordinates.csv`：配准后的全部坐标。
- `fluence.h5`：正向传播光通量，保存在会话目录。
- `Adot.nc` / `Adot.npz`：维度为 `(channel, vertex, wavelength)` 的灵敏度矩阵。
- `reconstruction.nc`：Cedalion `ImageRecon` 输出。
- `dot_project_manifest.json`：输入来源、参数和输出维度摘要。

## 重要边界

- MCX CUDA 需要 `pmcx` 和兼容的 NVIDIA/CUDA 环境；MCXCL 需要 `pmcxcl` 与 OpenCL；NIRFASTer 需要单独安装与当前 Python ABI 匹配的插件。界面会先运行 1,000 光子的隔离自检，只有实际模拟成功才启用按钮。
- 正向传播计算可能持续数分钟到数小时。当前 MVP 在 Web 进程内同步运行；生产部署应改为任务队列和独立计算 worker。
- 当前 Cedalion 正向模型代码中的多波长 fluence/光学参数处理仍带有 `FIXME`。在波长依赖和归一化完成独立验证前，输出适合算法开发，不应直接视为定量临床验证结果。
- 只有完成物理正向模型、Adot/Jacobian 和正则化逆解后才称为 DOT 重建；简单的通道值脑表面着色不是 DOT。
