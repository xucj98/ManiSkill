# 待办事项 (TODOLIST.md)

## 代码文档整理 *(P0)*

帮助Cursor更好地理解项目，提高后续的开发速度。

- [x] 创建 ARCHITECTURE.md：先搭建框架，列出计划中的主要目录和模块，简要说明其职责。随着重构深入，再逐步填充细节。
- [x] 创建 CONTRIBUTING.md：明确编码风格（如PEP8）、命名约定（类用CamelCase，函数/变量用snake_case），并强调"规范化提交"。
- [x] 修改README.md
- [x] 初始化 CHANGELOG.md：可以从你上次的 v0.1.0 开始，或者基于当前状态打一个新的有意义的标签。
- [x] 确定DEV_LOG.md和TODOLIST.md的内容
- [x] 编写Cursor Rules

## 基础设施与效率优化 *(P0)*

demo的存储方式作为基础设施需要优先实现，现在一个demo占用空间超过100GB，已经影响了后续大规模的实验。

- [x] **训练性能优化**
    - [x] 实现 JPEG 压缩 RGB 数据
        - [x] 在`odpc.data.demo.compress.py`中实现 RGB 图像的 JPEG 压缩
        - [x] 在`odpc.scripts.generate_demos.py`中增加数据集压缩的步骤
        - [x] 在`odpc.data.odpc_dataset.py`中支持读取压缩后的数据集
        - [x] 完成并通过对应的单元测试
        - [x] 在`odpc.scripts.vis.odpc_dataset.py`中增加数据集可视化的代码，用于验证`odpc_dataset`是否正确读取h5文件，其可以输出一个mp4视频
        - [x] 运行`odpc.scripts.generate_demos.py`和`odpc.scripts.vis.odpc_dataset.py`，验证正确性
        - [x] 在单元测试中对JPEG压缩率进行测试
    - [x] ~~研究并实现 ZFP 压缩 Depth 数据，在生成demo的时候增加压缩这一步骤，并在数据集读取的时候支持读取压缩的demo~~
        - [ ] ~~在`odpc.data.utils.py`中实现 ZFP 压缩和解压缩 depth 图像~~
        - [ ] ~~在单元测试中对depth图像的压缩率和MAE进行测试~~
        - [ ] ~~在`odpc.data.demo.compress.py`中增加对depth图像的压缩~~
        - [ ] ~~在`odpc.data.odpc_dataset.py`中支持读取压缩后的数据集~~
        - [ ] ~~完成并通过对应的单元测试~~
        - [ ] ~~在`odpc.scripts.vis.odpc_dataset.py`中增加深度图的可视化的代码，把深度图和rgb图片在宽度上concat一起显示就可以了~~
        - [ ] ~~运行`odpc.scripts.generate_demos.py`和`odpc.scripts.vis.odpc_dataset.py`，验证正确性~~


## 全面验证精度诅咒 *(P1)*

这是我们当前工作的核心发现，需要进一步充分地验证其有效性。我们需要验证其在不同任务/环境，不同观测，不同模型结构下都是有效的。

- [ ] 根据当前已经掌握的数据，编写精度诅咒的可视化代码，看到在固定成功率下，所需数据量随着精度要求增长的曲线。
- [ ] 在Peg Ins的任务下，测试不同观测和不同模型下的精度诅咒的有效性。
    - [ ] 修改Peg Ins的成功率评价：插入即成功，当前需要移动到指定位置才算成功，对于较大的洞口这两者有显著区别
    - [ ] 增加腕部相机
    - [ ] 学习tcp pose / object pose
    - [ ] 测试DINOv2作为backbone
- [ ] 在第二个任务上验证（细节待完善）


## A-IL 框架开发 *(P1)*

根据A-IL算法流程，完成整体架构开发。整体算法流程包括6个阶段，先在 `scripts/run_ail_cycle.py` 中完成整体框架设计，然后再对框架各个部分进行填充和实验。

- [ ] **架构设计与核心模块搭建**
    - [x] 完成 `ARCHITECTURE.md` 对 A-IL 架构的详细描述
    - [ ] 实现 A-IL 主循环脚本 (`scripts/run_ail_cycle.py`) 的基本框架
    - [ ] 定义核心数据结构 (例如，轨迹日志、高价值状态列表)
- [ ] **数据聚合与模型再训练 (Phase 1, 2, 5, 6)**
    - [ ] 在 `odpc.data.demo` 中增加 `generate_demo` 相关逻辑，支持随机初始化或根据给定的`initial_pose` 生成demo
    - [ ] 确保 `ImitationDataset` 支持数据聚合
    - [ ] 确保 `odpc.training.trainer` 可以被 A-IL 循环调用进行再训练
- [ ] **机器人自主探索与数据记录 (Phase 3)**
    - [ ] 实现策略在机器人上的自主部署与轨迹记录逻辑
    - [ ] 实现探索起点选择逻辑 (初始随机，后续基于高价值状态)
- [ ] **离线分析与高价值状态识别 (Phase 4)**
    - [ ] 实现失败/次优轨迹的自动定位模块
    - [ ] 集成模型不确定性估计算法 (例如基于 Ensemble)
    - [ ] 实现基于不确定性和/或影响力的状态排序
    - [ ] 实现多样性采样算法 (例如聚类) 以选择最终的高价值状态列表

## A-IL 实验与评估
- [ ] **数据集内生性度量 (贡献3)** *(P2)*
    - [ ] 实现基于预训练特征空间的数据集覆盖度/多样性度量
    - [ ] 实验验证该度量与策略性能的相关性
    - [ ] (可选) 将此度量集成到高价值状态识别流程中
- [ ] **A-IL 框架效果验证 (贡献2)** *(P1)*
    - [ ] 设计对比实验，比较 A-IL 与 Passive Learning, DAgger (或其他相关基线)
    - [ ] 在高精度任务上评估 A-IL 的数据效率和最终性能



## 可视化与调试 *(P2)* 
- [ ] **A-IL 流程可视化**
    - [ ] 可视化高价值状态的分布
    - [ ] (可选) 在 wandb 中记录 A-IL 每一轮的关键指标和可视化结果
- [ ] **失败案例分析**
    - [ ] 针对 A-IL 实验中的失败案例进行深入分析，指导后续改进
    - [ ] (可选) 实现一个基于真值的理想控制器用于对比分析
    - [ ] (可选) 实现一个简单的PID控制器来跟踪专家轨迹，作为性能下限参考
