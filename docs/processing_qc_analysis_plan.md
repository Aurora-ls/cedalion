# fNIRS 数据处理、质控与分析方案

本文档用于后续开发、实验设计和团队交流。目标不是规定唯一算法，而是建立一个可追溯、可比较、可扩展的处理框架：每一步都说明输入、输出、质量判据、参数来源和验证方法。

## 1. 目标与边界

### 1.1 目标

- 保留原始数据，任何清洗结果都可以回溯到原始测量。
- 将“坏通道”“坏时间段”“算法修复”和“统计建模”分开记录。
- 支持连续任务、事件相关任务、静息态、自然交互和 DOT 分析。
- 允许在同一份数据上比较多种处理链，而不是覆盖式修改。
- 让分析结果同时包含数值结果和 QC 证据，避免只报告最终 HbO/HbR 曲线。

### 1.2 不在本阶段解决的问题

- 不把某一组滤波截止频率当作所有设备、被试和任务的固定标准。
- 不把自动修复后的数据默认视为“真实信号”；严重运动段仍应支持 censoring 或剔除。
- 不把当前 DOT 输出表述为已经完成验证的全体积层析重建。

## 2. 数据契约

Cedalion 的核心数据对象是 `Recording`，时间序列使用带坐标和物理单位的 `xarray.DataArray`。

### 2.1 必须保留的对象

```text
rec
├── timeseries
│   ├── amp       原始光强/幅度，来源数据，不覆盖
│   ├── od        光学密度，可有多个版本
│   ├── conc      HbO/HbR 浓度变化，可有多个版本
│   └── ...       过滤、修复或模型残差等派生结果
├── masks         布尔质量掩码，True=clean，False=tainted
├── aux_obj       标量、阈值、统计量、算法元数据
├── aux_ts        辅助生理/触发/运动时间序列
├── stim          onset、duration、value、trial_type
├── geo3d/geo2d  探头几何和源-探测器坐标
└── meta_data     文件、设备、采样率、处理版本等元数据
```

### 2.2 轴与单位要求

- 时间轴必须包含 `time` 和 `samples`，优先使用秒作为统一计算单位。
- 通道轴必须保留 `channel`、`source`、`detector`；不得根据绘图顺序推断通道含义。
- 光强、OD、浓度使用不同的 timeseries key，避免同名覆盖。
- 所有处理函数应尽量保持 xarray 坐标和 pint 单位；进入 NumPy、统计模型或机器学习前，显式 `pint.dequantify()`。
- 采样率应由时间轴计算并记录；若采样间隔不均匀，应先重采样或停止使用依赖均匀采样的算法。

## 3. 推荐处理流水线

```text
读取 SNIRF/BIDS
    ↓
输入完整性检查与数据快照
    ↓
原始光强质量控制
    ├─ mean_amp / SNR / SCI / PSP
    ├─ source-detector distance
    └─ 设备丢帧、NaN、非正值、饱和检查
    ↓
生成并合并质量 mask，记录坏通道
    ↓
光强 → OD
    ↓
OD 域运动检测与处理
    ├─ id_motion / detect_outliers / GVTD
    ├─ TDDR 或 wavelet
    ├─ spline / spline_sg
    └─ 严重片段 censoring
    ↓
慢漂移和频率处理
    ├─ GLM 漂移回归优先
    └─ 必要时 freq_filter
    ↓
OD → HbO/HbR concentration
    ↓
短距离通道/全局生理噪声回归
    ↓
事件 epoch、block average、GLM、DOT 或数据驱动分析
    ↓
QC 报告、派生数据、参数和版本归档
```

实际项目中可以保留多个分支，例如 `od_raw`、`od_tddr`、`od_wavelet`、`conc_shortreg`，再使用相同的分析模型比较处理差异。

## 4. 各处理阶段的方案

### 4.1 输入完整性检查

检查内容：

- 文件是否可读，SNIRF/BIDS 元数据是否完整。
- `time` 是否单调、长度是否与数据一致、采样间隔是否稳定。
- 波长、通道、源、探测器坐标是否一一对应。
- 是否存在 NaN、Inf、非正光强、饱和、长时间零值或异常丢帧。
- `stim` 是否包含 `onset`、`duration`、`value`、`trial_type`。

输出：`input_integrity.json` 或等价的 `rec.aux_obj`，包含检查结果、警告和错误。

### 4.2 原始光强质控

现有函数：`cedalion.sigproc.quality`。

| 指标 | 作用 | 建议用途 |
|---|---|---|
| `mean_amp` | 发现过暗、饱和或量程异常通道 | 硬门槛或设备相关阈值 |
| `snr` | 粗略评估均值与波动关系 | 只作为辅助指标，不单独决定保留 |
| `sci` | 两波长心跳频段耦合质量 | 判断光-皮耦合 |
| `psp` | 心跳频段峰值功率 | 与 SCI 联合使用 |
| `sd_dist` | 源-探测器距离 | 区分短通道、长通道和几何异常 |

建议先生成独立 mask，再使用 `prune_ch` 合并。对探索性分析优先“标记为 NaN”，对明确坏通道再物理删除。每个 mask 都应保存阈值、单位和计算时间窗。

### 4.3 光强到 OD

使用 `cedalion.nirs.cw.int2od()`。转换前必须处理非正值；`repair_amp()` 可用于插值 NaN 和替换非正值，但应记录其使用，避免把传感器故障伪装成正常信号。

建议同时保存：

- 原始 `amp`；
- 修复输入后的 `amp_repaired`；
- `od`；
- 修复发生的样本 mask。

### 4.4 运动伪迹检测与处理

检测和修复应分开：

1. 先生成运动 mask 和时间段；
2. 再选择修复算法；
3. 修复后重新计算质量指标；
4. 对严重或持续运动段保留 censoring 结果。

现有方法：

- `id_motion`：基于短时间变化和幅值阈值；
- `detect_outliers`：异常点检测；
- `gvtd`：跨通道全局运动检测；
- `detect_baselineshift`：基线跳变检测；
- `tddr`：对瞬时大导数进行稳健修复；
- `wavelet`：对小波系数异常进行抑制；
- `spline` / `spline_sg`：对运动段进行样条/平滑修复；
- `pca_recurse`：对运动段使用 PCA 处理。

建议的默认比较顺序是 `TDDR` 与 `wavelet`；存在明显基线跳变时再比较 `spline_sg`。对全局运动强、连续时间长的记录，优先 censor，而不是无限叠加修复算法。

### 4.5 漂移、心跳和频率处理

`freq_filter()` 提供 Butterworth 滤波，但滤波不是唯一的漂移处理方案。事件相关 GLM 中优先使用 `drift_regressors`、Legendre 或 cosine regressors；只有在分析目标明确时才使用带通/低通滤波。

参数必须绑定到：

- 采样率；
- 实验类型（事件、block、静息）；
- 目标频带；
- 是否需要保留呼吸、心跳或快速反应。

所有滤波输出都要记录 `fmin`、`fmax`、阶数和边界处理方式。

### 4.6 生理噪声处理

优先级建议：

1. 有短距离通道：使用最近短通道、最大相关短通道或短通道平均回归量；
2. 有外部 PPG、呼吸、血压等辅助信号：作为 GLM 回归量或受约束 ICA 参考；
3. 没有辅助信号：使用全局均值/PCA，但必须做敏感性分析，防止误删真实全局神经信号。

现有接口包括 `closest_short_channel_regressor()`、`max_corr_short_channel_regressor()`、`average_short_channel_regressor()` 和 `global_component_subtract()`。

### 4.7 浓度转换

使用 `cedalion.nirs.cw.od2conc()`，必须记录：

- DPF 来源与数值；
- extinction spectrum；
- 源-探测器距离来源；
- 输出单位；
- 输入 OD 分支名称。

不同 DPF 或生理回归分支不要覆盖同一个 `conc` key，建议使用带处理标签的名称。

## 5. 分析方法

### 5.1 事件/Block 任务

推荐顺序：

1. 依据 `stim` 提取 epoch；
2. 负时间窗做基线校正；
3. 检查每个 epoch 的运动/质量 mask；
4. 先做 block average 进行可视化和快速 QC；
5. 再使用 GLM 估计条件效应和 HRF。

适合使用：

- `epochs.to_epochs()`；
- `hrf_regressors()` 或 `hrf_extract_regressors()`；
- `drift_regressors()`、Legendre/cosine 漂移项；
- 短通道回归量；
- `glm.fit()` 的 `ols`、`wls`、`gls`、`glsar`、`ar_irls`。

默认报告 HbO、HbR、条件 beta、标准误/置信区间、有效通道数和运动 censor 比例。

### 5.2 静息态/自然状态

可采用：

- 频带功率和心跳/呼吸相关性；
- 全局信号、短通道信号和通道间相关性；
- ICA、CCA、tCCA、SPoC/mSPoC；
- 网络/连接分析。

静息态分析必须报告是否去除了全局成分、短通道回归量和运动时间段，否则不同处理链之间不可比较。

### 5.3 DOT/空间分析

先在通道空间完成 QC、运动处理和浓度转换，再进入 `cedalion.dot`。空间重建前应记录：

- 使用的通道集合；
- measurement variance/covariance；
- 正则化方法和参数；
- 头模型、探头配准和表面来源；
- 缺失通道或坏通道如何处理。

### 5.4 机器学习

推荐流程：

1. 先按 epoch 划分训练/测试，避免同一试次或相邻时间泄漏；
2. 只在训练集拟合标准化、特征选择和降维；
3. 使用 `mlutils.features.epoch_features()` 生成 slope、mean、max、min、auc 等特征；
4. 使用 `mlutils.cv.create_cv_splits()` 或按被试/试次分组交叉验证；
5. 同时报告准确率、平衡准确率、混淆矩阵和跨被试泛化结果。

## 6. QC 输出规范

每次处理至少输出以下内容：

```text
qc/
├── input_integrity.json
├── channel_metrics.parquet/csv
├── channel_masks.nc 或等价文件
├── motion_segments.csv
├── processing_parameters.json
├── processing_log.jsonl
├── before_after_traces.png
├── scalp_quality.png
└── summary.md
```

### 6.1 必报指标

- 总通道数、保留通道数、删除通道数；
- 每个质量指标的分布和阈值；
- 每个通道的有效时间比例；
- 运动时间比例和最长运动段；
- 修复前后 SCI、PSP、SNR、GVTD；
- 各处理分支之间的 HRF/beta 差异；
- GLM 设计矩阵秩、条件数或共线性警告；
- 最终分析所使用的通道、时间段和参数版本。

## 7. 验证与比较设计

### 7.1 单元和回归测试

- 输入检查：时间轴、单位、坐标、NaN/非正值。
- 质量函数：阈值边界、mask 维度、通道删除与 NaN 标记。
- 运动算法：合成尖峰、基线漂移、连续运动段。
- 滤波：采样率、低通、高通、带通、单位保持。
- GLM：合成 HRF、漂移、自相关噪声和已知 beta 恢复。
- epoch：边界事件、不同采样率、缺失时间段。

### 7.2 算法比较指标

对每种处理链同时评估：

- 伪迹段残差下降；
- 干净段信号保真度；
- HRF 峰值、延迟和 AUC 偏差；
- GLM beta 恢复误差；
- 有效时间比例；
- 通道保留率；
- HbO/HbR 方向一致性；
- 跨被试或跨设备稳定性。

不能只用“曲线更平滑”作为算法优劣标准。

## 8. 后续开发分期

### P0：统一数据和 QC 契约

- 增加输入完整性检查。
- 统一 mask 命名、True/False 语义和元数据。
- 为所有派生 timeseries 记录来源、算法和参数。
- 补齐 py312 环境依赖后运行现有质量/运动/频率/GLM 测试。

### P1：可组合预处理流水线

- 新增配置对象，例如 `PreprocessingConfig`。
- 将质量、OD、运动、滤波、生理回归封装为可组合任务。
- 支持保留多条处理分支，不覆盖原始对象。
- 生成统一 QC summary 和处理日志。

### P2：验证与研究分析

- 建立合成伪迹 benchmark。
- 比较 TDDR、wavelet、spline_sg、PCA 和 censoring。
- 增加短通道/外部生理信号的回归实验。
- 增加 GLM 噪声模型和跨被试分析报告。

### P3：交互与发布

- 提供通道/时间段 QC 可视化。
- 支持导出处理后的 SNIRF/BIDS 及完整参数清单。
- 将方案中的稳定接口写入用户文档和示例 notebook。

## 9. 当前实现状态摘要

| 能力 | 当前状态 | 后续重点 |
|---|---|---|
| 质量指标与 mask | 已有函数和测试 | 统一配置、报告和阈值记录 |
| OD/浓度转换 | 已有 | 增加分支命名和元数据 |
| 运动检测/修复 | 多种算法已有 | 统一入口和 benchmark |
| 滤波 | 已有 Butterworth | 增加采样率/参数检查 |
| 短通道回归 | GLM 设计量已有 | 完善任务封装和比较报告 |
| GLM 自相关模型 | 已支持多种模型 | 增加模型选择和 QC 指标 |
| Epoch/block average | 已有 | 增加质量片段处理策略 |
| DOT | 已有通道到空间的分析能力 | 完善输入 QC 与重建证据链 |
| 自动 QC 报告 | 尚未统一 | P0/P1 优先 |
| 端到端预处理配置 | 尚未统一 | P1 优先 |

## 10. 开发约定

- 新算法先以独立函数和测试加入，再接入 Recording 级任务。
- 不删除原始数据；派生数据使用新 key。
- 任何自动删除、插值、回归或 censoring 都必须产生可检查的 mask 或日志。
- 参数默认值必须说明适用条件；设备或实验专属参数放在配置中，不写死在算法内部。
- 研究结论应区分“代码已有”“测试验证”“真实设备验证”和“待实验比较”。

