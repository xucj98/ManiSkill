# 更新日志 (Changelog)

所有此项目的值得注意的更改都将记录在此文件中，或通过 GitHub Releases 发布。

**对于 v0.5.0 及更早版本的详细变更，请参见本文档下方。**

**对于 v0.6.0 及更新版本的变更，请访问项目的 [GitHub Releases 页面]([你的项目仓库URL]/releases)。**


本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) 格式，并尽量遵循 [语义化版本 (Semantic Versioning)](https://semver.org/lang/zh-CN/)。

## [Unreleased] - 2025-06-23 

### Added
- ... (将当前未包含的最新更改放在这里)

### Changed
- ...

### Fixed
- ...

---

## [0.5.0] - 2025-06-17 (SOTA 探索与可复现性增强)

### Added
- 集成 `timm` 库，提供多样化的视觉编码器选项。 (`aa94791`)
- 引入混合精度训练 (`fp16`)，加速训练并减少显存占用。 (`aa94791`, `bdded85`)
- 训练时启用 `pin_memory` 提升数据加载效率。 (`d42ece6`)
- 全面增强实验可复现性：
    - 演示生成脚本 (`generate_demos.py`) 增加 `--seed` 参数。 (`5c8e77e`)
    - 生成演示数据时，自动记录所使用的配置文件。 (`9f32d9f`)
    - 训练和测试脚本现在会打印完整的实验配置。 (`cc70b7e`)
- 新增对演示数据中 RGB 图像进行 JPEG 压缩的功能，减小存储体积。 (`8bf78cb`)
- 引入 `wandb_group` 参数，并基于此组织实验输出目录 (`save_dir`)，便于实验管理。 (`f6757f4`, `0260c3a`)
- `vis_peg_pose` 脚本增加 headless 模式，支持在无显示服务器运行。 (`93e4cac`)
- 在 peg-in-hole 任务的演示生成中，增加了插入前的随机旋转，增强数据多样性。 (`a70494f`)
- 新增 `EeDeltaPose_TcpCrop128` 实验配置文件。 (`fc78c68`)

### Changed
- 重构 Policy 和 Agent 代码，引入基类，并支持直接学习末端执行器相对位姿 (EeDeltaPose)。 (`219843d`)
- 调整了 EMA (Exponential Moving Average) 的默认参数。 (`690ecff`)
- 调整了训练脚本的默认学习率 (`lr`) 为 `1e-3`。 (`71fd15d`)
- 调整了部分训练的默认参数。 (`4dc189f`)
- 调整了部分配置文件：`odpc` 默认不再应用 TCP crop 观测预处理。 (`7ed899a`)

### Fixed
- 修复了因代码重构（如引入 Policy/Agent 基类）后 `valid.py` 脚本运行不正确的问题。 (`8677626`)

---

## [0.4.0] - 2025-06-06 (高级数据处理与评估体系完善)

### Added
- 实现 `TCP Crop` (Tool Center Point 局部裁剪) 观测预处理流程，支持 Tensor 输入和批处理。 (`b87e4ff`, `eb9e0bb`, `0a78945`)
- `ODPCDataset` 增加 `clip_traj` 参数，实现对演示轨迹的按需截断功能。 (`de55f0b`, `95180c9`)
- 实验配置 (`config.yaml`) 现在会自动保存到实验输出目录。 (`a407a25`)

### Changed
- `train.py` 现在使用统一的 `Evaluator` 类进行模型评估。 (`d6f1786`)
- 评估环境 (`eval_env`) 现在输出原始的多层结构化观测，不再默认使用 `FlattenRGBDObservationWrapper`。 (`502f258`)
- 调整了 PegInsertion 任务中 Box 的初始位置，以确保洞口始终可见。 (`551d2ac`)
- 优化了预测结果的可视化效果。 (`d4e247b`)

### Fixed
- **关键修复**: 解决了在模型训练时评估未使用 EMA (Exponential Moving Average) 模型的问题。 (`b63a2a5`)
- 修复了因文件夹名称包含日期导致视频保存路径错误的问题。 (`696dc3c`)
- 修复了评估 (`evaluate.py`) 时会额外生成一个无内容的预测视频的问题。 (`ccedf6c`)
- 移除了动作空间定义中的 NaN 值，以更好兼容 CPU 仿真。 (`3ffd060`)
- 修复了 `valid.py` 在评估结束后未关闭环境的问题。 (`9072c46`)

---
## [0.3.0] - 2025-05-29 (项目结构化与可维护性革命)

### Added
- 引入 `setup.py`，项目正式打包为可安装的 Python 包。 (`4751abe`)
- 新增 `requirements.txt` 用于管理项目依赖，`TODOLIST.md` 用于项目跟踪。 (`b4b3805`)
- 创建初版项目文档，包括 ODPC 项目概述、技术原理、以及论文相关的摘要、引言等草稿。 (`a840622`)
- 新增 `generate_demos.py` 脚本和 `peg_insertion_demo.yaml` 配置文件，用于生成 PegInsertion 任务的演示数据。 (`5c2d4c9`)
- 新增 `peg_pose.py` 和 `visualize.py` 工具，用于可视化相机坐标系下的物体位姿。 (`e33fc27`)
- `generate_demos.py` 脚本现在支持通过 `env_kwargs` 设置环境初始化参数。 (`f387ea7`)
- 实现了模型在数据集上的离线评估功能。 (`7ad331a`)

### Changed
- **重大重构**: 对整个项目的文件结构进行了调整，核心代码封装到 `odpc` 包内，相关脚本移至 `scripts/` 目录，配置文件移至 `configs/` 目录。 (`4751abe`)
- 调整了部分模块的组织，例如将 `visualize_pose` 函数移至 `odpc.utils.visualize`。 (`02e1388`)
- 早期文件结构调整，将 `peg_insertion_side_v2` 环境和其 motion planning solver 移入 `odpc` 目录。 (`1c67f9c`)

### Fixed
- 修复了早期训练脚本和 WandB 配置中的一些错误。 (`8bd8cb8`, `f47547c`)
- 修复了一个代码中的拼写错误。 (`c412862`)

---

## [0.2.0] - 2025-05-16 (ODPC 核心逻辑与数据流实现)

### Added
- 完整实现了 `ODPCDataset`，用于加载和处理模仿学习数据。 (`722171d`, `510242e`)
- 实现了 `DataConversion` 模块，负责从原始数据计算模型预测目标，以及从模型预测计算机器人控制量。 (`d72f5dd`)
- 实现了 ODPC 核心训练流程 (`e2160eb`) 和在仿真环境中的验证 (`evaluate`) 流程 (`1be8c87`, `ba27bcb`)。
- 支持 `pd_ee_pose` 控制模式，并实现了对预测结果的可视化。 (`7643452`, `d0a29d6`)
- 为 XArm6 机械臂实现了 PegInsertion 任务的 motion planning 策略。 (`10b82e0`)
- 在 ODPC 训练中增加了对 OOD (Out-of-Distribution) 环境的测试。 (`0e43c34`)
- 新增在数据集上评估 ODPC 模型性能的功能 (`evaluate_odpc_on_dataset`)。 (`5585ca9`, `c98e970`)
- 相机增加了多种模式 (`fixed`, `random`, `move`) 和特定视角 (`fixed-2`)。 (`4d07bac`, `0b4a40a`)
- 在 motion planning 策略中增加了随机性，抓取位置进行了随机化。 (`6e46f43`, `01bcdf5`)
- 绘制了 `peg_pos_z` 和 `delta_pose` 曲线用于数据分析。 (`98885cc`)
- 测试了 ODPC 的数据处理流程，包括坐标变换等。 (`a54918e`)

### Changed
- `ODPCAgent` 的输出现在支持整个预测时域 (`pred_horizon`)，具体截取在 Wrapper 中处理。 (`3dbd5e1`)
- 在DP训练中增加了 `depth_clamp`, `used_cameras`, `use_state` 的参数。 (`bf531ce`)

### Fixed
- 修复了数据处理时 `depth` 数据丢失的 Bug (多次提交)。 (`e436b2e`, `46155c6`)
- 修复了因 `inplace` 操作导致 `pred_to_visualize` 可视化出错的 Bug。 (`aa6522b`)
- 修复了 EMA (Exponential Moving Average) 模型在评估时缺少 BN (Batch Normalization) 统计值的问题。 (`5585ca9`)

---

## [0.1.0] - 2025-05-01 (项目奠基与首次端到端成功)

### Added
- 成功实现了基于 RGBD 输入的 PegInsertionSide 任务的端到端训练。 (`347cb88`)
- 新增 `PegInsertionSideV2` 任务环境。 (`30d8276`)
- 初步的 `train_rgbd.py` 训练脚本。
- `common` 模块增加了 debug 用的 log 函数。 (`f509d7f`)
- 新增 `analyze_h5.py` 工具，用于打印 HDF5 文件结构。 (`f509d7f`)
- 新增 `h5_to_gflow.py` 脚本 (用途需进一步明确)。 (`a34c3e1`)

### Changed
- 优化了 `diffusion_policy` 的 `Dataset` 实现（在 `train_rgbd.py` 中）：
    - 避免一次性加载整个 HDF5 文件到内存。
    - 避免在 `__init__` 中将数据移到 CUDA 设备。
    - 在 `__getitem__` 中独立打开 HDF5 文件句柄。 (`df7989e`)
- 优化了 `trajectory_replay.py` 中 HDF5 图片存储的 `chunks` 设置。 (`df7989e`)

### Fixed
- 修复了观测空间顺序与 HDF5 文件中 `key` 顺序不一致可能导致的问题。 (`0380aab`)
- `train_rgbd.py` 中 `to_device(data)` 改为使用 `common.to_tensor(data, device)`。 (`6fe7b1c`)
- `train_rgbd.py` 中的 `assert` 被注释掉（早期调试）。 (`3a7ab57`)
- `peg_insertion_side_v2.py` 修正了相机模式判断逻辑。 (`3a7ab57`)

---