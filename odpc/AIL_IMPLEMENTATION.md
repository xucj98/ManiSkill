# A-IL 框架实现细节 (AIL_IMPLEMENTATION.md)

本文档详细阐述了 **A-IL (Active Imitation Learning)** 框架中各个核心模块的具体实现方法、接口定义和数据结构规范。它是对 `ARCHITECTURE.md` 中高层设计的技术补充，旨在为开发工作提供清晰、可执行的指南。

## 1. 核心数据结构

A-IL 框架的稳定运行依赖于两个核心数据结构在不同阶段之间的正确流转。

### 1.1. 轨迹日志 (`TrajectoryLog`)

-   **用途**: 在 A-IL **阶段3 (策略自主部署)** 中，由 AI Agent 与环境交互产生，作为 **阶段4 (离线分析)** 的输入。
-   **格式**: **完全复用现有的专家演示 (demo) 的 HDF5 文件格式**。这最大化了代码的复用性，使得处理轨迹日志的工具可以与处理专家演示的工具保持一致。
-   **实现要点**:
    -   在 A-IL 策略部署阶段，轨迹日志将通过我们自定义的 `AILRecordEpisode` 环境包装器生成。`AILRecordEpisode` 继承自 `mani_skill.utils.wrappers.record.RecordEpisode`，并扩展了其功能。
    -   `AILRecordEpisode` 将负责合并 `agent.get_action()` 返回的 `info_dict`（包含 `model_uncertainty` 等）与 `env.step()` 返回的 `info` 字典，并将其存储到 HDF5 轨迹文件中。
    -   与标准专家演示的关键区别在于，A-IL 策略部署时录制的 `info` 字典中，**必须额外包含** 以下用于离线分析的关键信息：
        -   `model_uncertainty`: 模型对当前状态所做决策的不确定性度量。如果使用 Ensemble 模型，这可以是多个模型输出之间的方差或标准差。
        -   `raw_action`: 策略网络输出的原始动作，在经过任何后处理（如 action processor）之前的值。
        -   其他有助于调试和分析的中间值。
    -   因此，在实现 `policy_rollout` 函数时，需要确保 Agent 的 `get_action` 方法返回的 `info_dict` 包含这些信息，并且 `AILRecordEpisode` 能够正确地将其存入 HDF5 文件的 `info` 数据集中。

### 1.2. 高价值状态列表 (`HighValueStateList`)

-   **用途**: 作为 A-IL **阶段4 (离线分析)** 的输出，和 **阶段5 (靶向性专家演示)** 的输入。它为专家指明了需要在哪些“弱点”或“难点”上进行纠正性演示。
-   **格式**: 一个 Python `list` 对象，通过 `pickle` 序列化后保存为 `.pkl` 文件。
    -   列表中的每一个元素都是一个**完整的、可用于重置环境的环境状态**。
    -   这个“状态”对象是通过调用环境的 `env.get_state()` 方法获得的，它包含了恢复场景所需的一切信息（例如，在 Peg-in-Hole 任务中，这包括机械臂的关节状态、Peg 的位姿、目标孔洞的位姿等）。
-   **实现要点**:
    -   **离线分析 (`offline_analysis`)**: 该模块的核心任务之一就是从输入的 `TrajectoryLog` 中，根据失败定位、不确定性排序等算法，在轨迹的关键时间点（例如，失败前的某个时刻）提取出对应的环境状态。前提是 `TrajectoryLog` 中必须保存了每一步的环境状态。
    -   **靶向性演示 (`generate_targeted_demos`)**: 该模块将加载 `.pkl` 文件，反序列化得到状态列表。然后，它会遍历这个列表，在每次循环中调用 `env.set_state()` 方法将环境精确重置到指定状态，之后再启动专家演示的录制流程。

## 2. 核心模块实现规划

本节将随着开发进度，逐步填充 `scripts/run_ail_cycle.py` 中每个阶段函数的具体实现思路。

### 2.1. `generate_expert_demos` (生成专家演示)

-   **核心思路**: 根据 `initial_states_path` 是否为 `None`，决定是生成初始演示还是靶向性演示。
-   **输入**: 实验配置 (`config`)、演示数据集的保存目录 (`output_dir`)、可选的高价值状态列表路径 (`initial_states_path: Optional[pathlib.Path]`)。
-   **输出**: 生成的演示数据集的路径 (`demo_path`，HDF5 格式)。
-   **实现要点**:
    1.  **初始演示**: 如果 `initial_states_path` 为 `None`，则调用 `scripts/generate_demos.py` (或其核心逻辑) 来生成初始演示。演示数量由 `config.initial_demo_generation.num_traj` 控制。
    2.  **靶向性演示**: 如果 `initial_states_path` 不为 `None`，则加载 `initial_states_path` 中包含的高价值状态列表。**每个高价值状态只生成一个专家轨迹**。因此，靶向性演示的总数将等于高价值状态的数量，这个数量由 `config.ail_loop.targeted_demos_per_cycle` (与 `config.offline_analysis.num_high_value_states` 保持一致) 控制。

### 2.2. `train_policy` (策略训练)

-   **核心思路**: 调用 `odpc.training.trainer` 中的核心训练逻辑。
-   **输入**: 实验配置 (`config`)、**数据集路径列表 (`dataset_paths: List[pathlib.Path]`)**、输出目录 (`output_dir`)。
-   **输出**: 训练好的模型检查点路径 (`policy_ckpt_path`)。
-   **实现要点**:
    1.  **数据集加载**: `odpc.data.odpc_dataset.ImitationDataset` 需要支持加载一个包含多个 HDF5 文件路径的列表，并能够将它们聚合为一个统一的数据集进行训练。
    2.  **训练逻辑**: 调用 `odpc.training.trainer` 中的训练函数或类，传入配置和聚合后的数据集。**在训练过程中，将同步进行评估，并保存表现最好的模型检查点。** 评估的频率和保存策略将由 `config.trainer.eval_freq` 和 `config.trainer.save_freq` 控制。

### 2.3. `policy_rollout` (策略部署)

-   **核心思路**: 利用 `odpc.odpc.evaluation.evaluate.py` 中的 `evaluate_on_env` 函数来驱动策略与环境的交互。**记录功能将完全由环境包装器 `AILRecordEpisode` 处理**。
-   **输入**: 策略模型检查点路径 (`policy_ckpt_path`)、实验配置 (`config`)、当前轮次产物目录 (`cycle_dir`)。
-   **输出**: 轨迹日志文件路径 (`rollout_log_path`，HDF5 格式)。
-   **实现要点**:
    1.  **加载策略与智能体**: 根据 `policy_ckpt_path` 加载训练好的策略模型，并实例化 `BaseAgent`。
    2.  **环境实例化与包装**: **通过一个环境工厂函数（可能参考 `odpc/odpc/envs/factory/make_eval_envs.py` 的模式）来创建并包装环境**。这个工厂函数将确保环境被 `AILRecordEpisode` 包装器正确应用，并配置好轨迹保存路径和文件名。
        -   **起始状态选择**: 在 A-IL 框架的初期实现中，策略部署的起始状态将**采用随机初始化**。后续在框架稳定并能达到较高成功率后，再考虑引入基于高价值状态的智能引导。
        -   `AILRecordEpisode` 将负责合并 `agent.get_action()` 返回的 `info_dict`（包含 `model_uncertainty` 等）与 `env.step()` 返回的 `info` 字典，并将其存储到 HDF5 轨迹文件中。
    3.  **调用 `evaluate_on_env`**: 将实例化并包装好的环境以及加载的智能体传递给 `odpc.odpc.evaluation.evaluate.evaluate_on_env` 函数。`evaluate_on_env` 将驱动智能体在环境中执行指定数量的 episode（数量由 `config.ail_loop.rollout_episodes_per_cycle` 控制）。
    4.  **轨迹保存**: `AILRecordEpisode` 包装器将在部署结束后自动将轨迹数据（包括合并后的 `info`）保存到指定的 HDF5 文件中。`policy_rollout` 函数将返回这个 HDF5 文件的路径。

### 2.4. `offline_analysis` (离线分析)

-   **核心思路**: 分析部署日志，识别高价值状态。
-   **输入**: 实验配置 (`config`)、当前轮次产物目录 (`cycle_dir`)、策略部署阶段产出的轨迹日志文件路径 (`rollout_log_path`)。
-   **输出**: 识别出的高价值状态列表的路径 (`high_value_states_path`，`.pkl` 格式)。
-   **实现要点**:
    1.  **加载轨迹日志**: 从 `rollout_log_path` (HDF5 文件) 中读取轨迹数据，特别是 `env_states` 和 `info` 字典（包含 `model_uncertainty`）。
    2.  **筛选失败/次优轨迹**: 根据 `config.offline_analysis.failure_threshold` 等参数，识别出策略表现不佳的轨迹。
    3.  **不确定性/影响力排序**: 利用 `config.offline_analysis.uncertainty_metric` 指定的方法，对失败轨迹中的状态进行排序，找出最有学习价值的状态。
    4.  **采样一个多样化的高价值状态集合**: 根据 `config.offline_analysis.diversity_sampling_method` 和 `config.offline_analysis.num_high_value_states`，从排序后的状态中采样一个多样化的子集。**`config.offline_analysis.num_high_value_states` 应该与 `config.ail_loop.targeted_demos_per_cycle` 保持一致，因为每个高价值状态只生成一个专家轨迹。**
    5.  **保存高价值状态**: 将采样到的环境状态序列化为 `.pkl` 文件。

---
*本文档将随着开发的深入而持续更新。*
