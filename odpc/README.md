# A-IL: 通过主动模仿学习克服机器人操作中的精度诅咒 (Active Imitation Learning to Overcome the Curse of Precision in Robotic Manipulation)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
<!-- 可选：如果你有许可文件 -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- 可选：如果你有项目主页或论文链接 -->
<!-- [![Project Page](https://img.shields.io/badge/Project-Page-green)](YOUR_PROJECT_PAGE_LINK) -->
<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv%3Axxxx.xxxx-red)](YOUR_PAPER_LINK) -->

## 📜 项目概述 (Overview)

本项目旨在解决机器人模仿学习在推广至具有工业价值的高精度、接触丰富的操作任务时所面临的**数据效率瓶颈**。我们引入并系统研究了 **“精度诅咒 (Curse of Precision)”** —— 一个普遍存在的规律，即所需的专家演示数据量随任务精度要求呈指数级增长。

为了克服这一挑战，我们提出了 **A-IL (Active Imitation Learning)**，一个新颖的“批量化失败引导的主动学习”框架。A-IL 通过智能地引导机器人探索、离线分析失败数据并请求针对性的批量专家演示，旨在显著提升在高精度任务上的数据采集效率和策略性能。

本项目遵循一个“发现问题 -> 量化规律 -> 提出通用解法 -> 提供评估工具”的完整研究叙事。

## ✨ 核心贡献与特性 (Core Contributions & Features)

1.  **系统性发现“精度诅咒” (Systematic Discovery of the "Curse of Precision")**:
    *   首次将数据规模定律 (Scaling Law) 的研究从传统的泛化性维度扩展到任务精度维度。
    *   通过实验定量揭示了在高精度操作模仿学习中，所需演示数据量随任务精度要求呈指数级增长的普遍规律。
    *   为理解模仿学习在高精度应用上的瓶颈提供了关键的理论基石。

2.  **A-IL 框架 (A General-Purpose Active Learning Framework - A-IL)**:
    *   提出一个“批量化、周期性、恢复导向”的主动学习范式，以高效解决“精度诅咒”。
    *   **核心流程**:
        1.  **机器人探索 (Robot Rollout)**: 使用当前策略进行靶向性探索，主动发现薄弱环节和失败模式。
        2.  **离线分析 (Offline Analysis)**: 智能分析探索轨迹，生成“高价值演示请求列表”。
        3.  **批量专家演示 (Batched Human Demonstration)**: 根据请求列表，进行集中的、批量的、有针对性的数据采集。
    *   将人机交互从“实时微观”提升到“周期宏观”，显著降低上下文切换成本。

3.  **数据集内生性度量 (Novel Intrinsic Metric for Dataset Quality)**:
    *   提出一个无需训练即可评估数据集质量的内生性度量，基于数据在预训练特征空间中的多样性与覆盖度。
    *   可用于指导 A-IL 中更高效的数据采集，目标是主动优化数据集质量。

## 🚀 快速开始 (Getting Started)

### 1. 环境搭建 (Environment Setup)

-   克隆本仓库：
    ```bash
    git clone [你的项目仓库URL]
    cd [项目目录名称]
    ```
-   (推荐) 创建并激活 Conda 虚拟环境：
    ```bash
    conda create -n ail_env python=3.8
    conda activate ail_env
    ```
-   安装依赖：
    ```bash
    pip install -r requirements.txt
    # 如果您计划修改 odpc 包（未来可能重命名为 ail_project），并希望更改立即生效：
    pip install -e .
    ```

### 2. 项目结构概览 (Project Structure)

```
.
├── ARCHITECTURE.md         # 项目架构详解
├── CHANGELOG.md            # 版本更新日志
├── CONTRIBUTING.md         # 贡献指南
├── README.md               # 你正在阅读的文件
├── configs/                # 实验和演示的配置文件
├── odpc/                   # 核心 Python 包 (未来可能重命名)
│   ├── data/               # 数据加载、处理、增强、数据集质量度量
│   ├── envs/               # 仿真环境与装饰器
│   ├── evaluation/         # 策略评估与指标计算
│   ├── models/             # Agent、Policy、视觉编码器模型
│   ├── training/           # 核心策略训练逻辑
│   └── utils/              # 通用工具函数
├── progress/
│   ├── DEV_LOG.md          # 开发日志 (可选，更多细节可能在外部知识库)
│   └── TODOLIST.md         # 当前任务列表
├── scripts/                # 驱动型脚本 (训练、评估、数据生成、A-IL循环)
├── requirements.txt        # Python 依赖
└── setup.py                # 打包脚本
```
更详细的架构说明请参见 [ARCHITECTURE.md](ARCHITECTURE.md)。

### 3. 运行核心工作流 (Running Core Workflows)

(请注意：具体的脚本参数和运行方式请参考脚本本身的文档字符串或使用 `--help` 获取最新信息。部分脚本可能需要预先准备数据集或配置文件。)

-   **生成初始专家演示数据**:
    ```bash
    python scripts/generate_initial_demos.py --config configs/demo/YourTask_InitialDemos.yaml
    ```
-   **训练一个基础的模仿学习策略**:
    ```bash
    python scripts/train_policy.py --config configs/exps/YourTask_BC_Baseline.yaml
    ```
-   **评估已训练的策略**:
    ```bash
    python scripts/evaluate_policy.py --config configs/exps/YourTask_BC_Baseline.yaml --ckpt-path path/to/your/model.pth
    ```
-   **运行 A-IL 主动学习循环**:
    ```bash
    python scripts/run_ail_cycle.py --config configs/exps/YourTask_AIL.yaml
    ```
-   **(可选) 量化“精度诅咒”实验**:
    (此部分可能需要特定的脚本或多个训练/评估步骤，请参考相关文档或脚本说明)

更多详细的教程和使用示例，请参见 `docs/tutorials/` 目录 (待创建)。

## 📖 文档 (Documentation)

-   [**ARCHITECTURE.md**](ARCHITECTURE.md): 详细描述了项目的软件架构、模块划分和核心设计模式。
-   [**CONTRIBUTING.md**](CONTRIBUTING.md): 如何为本项目做出贡献，包括编码风格、分支策略和提交规范。
-   [**CHANGELOG.md**](CHANGELOG.md): 项目的版本历史和重要变更记录 (v0.5.0及之前)。
-   **GitHub Releases**: 对于 v0.6.0 及更新版本的详细变更，请访问项目的 [Releases 页面]([你的项目仓库URL]/releases)。
-   **(未来) `docs/tutorials/`**: 包含更详细的使用教程和示例。

## 🛠️ 如何贡献 (How to Contribute)

我们欢迎各种形式的贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 以了解详细的贡献流程和规范。

主要步骤包括：
1.  Fork 本仓库。
2.  创建您的特性分支 (`git checkout -b feat/YourAmazingFeature`)。
3.  提交您的更改 (`git commit -m 'feat(scope): Add some amazing feature'`)。
4.  将您的更改推送到分支 (`git push origin feat/YourAmazingFeature`)。
5.  发起一个 Pull Request。

## 📝 引用 (Citation)

如果您的研究从本项目中受益，请考虑引用我们的工作 (如果适用，请在此处添加您的论文引用信息)：

```bibtex
@article{YourLastNameYYYYail,
  title   = {A-IL: Active Imitation Learning to Overcome the Curse of Precision in Robotic Manipulation},
  author  = {Your Name and Co-authors},
  journal = {arXiv preprint arXiv:xxxx.xxxx (or Conference/Journal Name)},
  year    = {YYYY}
}
```

## 📄 许可证 (License)

本项目采用 [MIT License](LICENSE) (或其他您选择的许可证)。

---

我们希望这个项目能为机器人学习领域，特别是在高精度操作任务的数据效率方面，提供有价值的见解和工具。如果您有任何问题或建议，请随时通过 GitHub Issues 与我们联系。