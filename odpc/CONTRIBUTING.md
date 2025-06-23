# 贡献指南 (CONTRIBUTING.md)

欢迎对本项目做出贡献！我们非常感谢您的时间和精力。

本指南旨在帮助您顺利地为项目做出贡献。请花几分钟时间阅读，以确保您的贡献过程顺畅高效。

在开始之前，建议您先阅读项目的 [README.md](README.md) 以了解项目背景和目标，以及 [ARCHITECTURE.md](ARCHITECTURE.md) 以熟悉项目的整体架构。

## 如何开始

### 1. 设置开发环境

-   **先决条件**:
    -   Git
    -   Python (建议版本 >= 3.8)
    -   (推荐) Conda 用于管理 Python 环境，或您熟悉的任何其他虚拟环境工具 (如 venv, Poetry)。
-   **获取代码**:
    ```bash
    git clone [你的项目仓库URL]
    cd [项目目录]
    ```
-   **创建并激活虚拟环境 (以 Conda 为例)**:
    ```bash
    conda create -n your_env_name python=3.8
    conda activate your_env_name
    ```
-   **安装依赖**:
    ```bash
    pip install -r requirements.txt
    # 如果您计划修改 odpc 包本身，并希望更改立即生效，可以使用开发模式安装：
    pip install -e .
    ```
-   **(可选) 运行测试**:
    ```bash
    # (如果项目配置了 pytest 或其他测试运行器)
    # pytest
    ```
    请确保所有测试通过，以验证您的环境设置正确。

### 2. 理解和运行 `scripts/` 目录下的核心脚本

`scripts/` 目录包含了驱动本项目核心工作流程的 Python 脚本，例如模型训练、策略评估、数据生成等。在成功设置开发环境后，尝试运行这些脚本是熟悉项目的好方法。

**a. 脚本概览**:

-   `scripts/train_policy.py`: 用于独立训练一个策略模型 (例如，用于“精度诅咒”研究或调试特定模型)。
-   `scripts/run_ail_cycle.py`: 执行完整的 A-IL 主动学习循环。
-   `scripts/evaluate_policy.py`: 用于评估已训练好的策略模型。
-   `scripts/generate_initial_demos.py`: 用于生成初始的专家演示数据集。
-   `scripts/generate_targeted_demos.py`: 用于根据A-IL分析结果生成靶向性专家演示。
-   (根据你的实际情况列举其他重要脚本)

**b. 运行示例**:

在运行脚本之前，请确保：
-   您的虚拟环境已激活。
-   必要的配置文件 (位于 `configs/` 目录) 已准备好或您知道如何指定它们。
-   所需的数据集 (例如，初始演示数据) 已按预期放置 (参考 [tutorials.md(todo)](docs/tutorials.md) 中关于数据目录的说明)。

**示例：运行策略评估**
(假设 `evaluate_policy.py` 需要一个模型检查点路径和一个配置文件)
```bash
python scripts/evaluate_policy.py --config configs/exps/MyExperiment.yaml --ckpt-path runs/MyExperiment/checkpoints/model_best.pth
```

**(可选) c. 调试提示**:

-   大多数脚本支持 `--help` 参数来查看其可用的命令行选项。
-   仔细阅读脚本的输出日志，它们通常包含重要的运行信息和可能的错误提示。
-   如果遇到问题，请检查您的配置文件和数据路径是否正确。

**注意**: 具体的脚本参数和运行方式可能会随着项目的迭代而变化。请参考脚本本身的文档字符串或使用 `--help` 获取最新的使用信息。

## 贡献方式

### 1. 报告 Bug

如果您在项目中发现了 Bug：

1.  请首先在项目的 [GitHub Issues]([你的项目仓库URL]/issues) 中搜索是否已有人报告过该 Bug。
2.  如果找不到相关的 Issue，请创建一个新的 Issue。
3.  在您的 Bug 报告中，请尽可能详细地提供以下信息：
    -   **清晰的标题和描述**。
    -   **复现 Bug 的详细步骤**。
    -   **您期望看到的行为**。
    -   **实际发生的行为**（包括任何错误信息和完整的堆栈跟踪）。
    -   **您的环境信息**（操作系统、Python 版本、`odpc` 包版本（如果有）、相关依赖库版本）。

### 2. 提出功能建议

如果您有关于新功能或改进现有功能的建议：

1.  请首先在项目的 [GitHub Issues]([你的项目仓库URL]/issues) 中搜索是否已有类似的建议。
2.  如果找不到相关的 Issue，请创建一个新的 Issue 来讨论您的建议。
3.  在您的建议中，请阐述：
    -   **您希望实现的功能**。
    -   **该功能试图解决什么问题或带来什么好处**。
    -   **(可选) 您对如何实现该功能的初步想法**。

### 3. 提交代码贡献 (Pull Requests)

我们非常欢迎代码贡献！请遵循以下步骤和规范，以确保您的贡献能够顺利地被整合：

#### a. 工作流程总览

1.  **创建特性分支**: 所有代码更改都应在从最新的 `main` (或 `develop`) 分支创建的**特性分支 (feature branch)** 上进行。
2.  **本地开发与提交**: 在您的特性分支上进行开发。您可以根据开发进度进行多次提交，这些提交主要用于记录您的工作过程。
3.  **准备 Pull Request**: 当一个功能单元开发完成、Bug 修复完毕或达到一个可审查的阶段时，准备发起一个 Pull Request (PR) 将您的特性分支合并到 `main` 分支。
4.  **发起 Pull Request**: 确保您的 PR 标题和描述清晰、规范。
5.  **代码审查与讨论**: 项目维护者或其他贡献者将对您的 PR 进行审查。请积极参与讨论并根据反馈进行修改。
6.  **合并 PR**: 一旦 PR 通过审查并满足所有要求，它将被合并。

#### b. 分支策略

-   从最新的 `main` 分支创建您的特性分支：
    ```bash
    git checkout main
    git pull origin main
    git checkout -b <分支名称>
    ```
-   **分支命名规范**: 请使用能清晰描述分支用途的名称，例如：
    -   `feat/add-new-metric` (新功能)
    -   `fix/resolve-issue-15` (修复 Bug，关联 Issue 编号)
    -   `docs/update-architecture-md` (文档更新)
    -   `refactor/optimize-data-loading` (代码重构)

#### c. 编码风格

-   **PEP 8**: 请遵循 [PEP 8 Python 代码风格指南](https://www.python.org/dev/peps/pep-0008/)。
-   **代码格式化**: (可选，但推荐) 考虑使用诸如 [Black](https://github.com/psf/black) 进行代码格式化，[Flake8](https://flake8.pycqa.org/en/latest/) 进行代码检查。
-   **命名约定**:
    -   类名：`CamelCase` (例如 `MyAwesomeClass`)
    -   函数名、方法名、变量名：`snake_case` (例如 `calculate_value`, `is_valid`)
    -   常量：`UPPER_SNAKE_CASE` (例如 `MAX_ITERATIONS`)
-   **文档字符串 (Docstrings)**:
    -   所有公开的模块、类、函数和方法都应有清晰的文档字符串。
    -   (推荐) 遵循 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) 或 [NumPy/SciPy Docstring Standard](https://numpydoc.readthedocs.io/en/latest/format.html) 中的一种。
    -   Docstrings 应解释代码的**目的、参数、返回值以及任何可能抛出的异常**。

#### d. 关于提交 (Commits) 与 Pull Requests (PRs) 的信息规范

我们区分对待特性分支上的日常开发提交和最终用于合并的 Pull Request 的信息规范。

**i. 特性分支上的开发提交 (Development Commits on Feature Branches):**

-   **目的**: 这些提交主要用于记录您在特性分支上的开发进度、保存阶段性成果或方便您个人回溯。
-   **灵活性**: 您可以根据需要频繁提交。这些提交的 message 可以相对简洁，例如：
    -   `WIP: Initial setup for data processing module`
    -   `Add basic anet class structure`
    -   `Refactor: Extract utility function for normalization`
    -   `Fix: Correct off-by-one error in loop (intermediate fix)`
-   **要求**:
    -   虽然灵活，但仍鼓励您编写有意义的、能大致描述所做更改的 commit message，即使只是为了方便自己回顾。
    -   **这些特性分支上的单个 commit message 不会直接成为最终 Changelog 的内容。**

**ii. Pull Request (PR) 的标题与描述:**

-   **目的**: PR 是将您的贡献整合到项目主干的关键步骤。PR 的标题和描述是代码审查、历史追溯以及**生成项目更新日志 (Changelog) 的重要依据**。
-   **PR 标题 (Title) - 必须遵循规范化提交 (Conventional Commits) 格式**:
    **格式**: `type(scope): subject`
    -   **`type`**: 描述提交的类型，必须是以下之一：
        -   `feat`: 引入新功能。
        -   `fix`: 修复 Bug。
        -   `docs`: 仅修改文档。
        -   `style`: 代码风格调整（不影响代码含义）。
        -   `refactor`: 代码重构。
        -   `perf`: 提升性能的代码更改。
        -   `test`: 添加或修正测试。
        -   `build`: 影响构建系统或外部依赖的更改。
        -   `ci`: 更改 CI 配置文件和脚本。
        -   `chore`: 其他不修改源代码或测试文件的变动。
    -   **`scope` (可选)**: 描述本次 PR 影响的范围/模块。例如：`agent`, `training`, `data`, `ail-cycle`, `docs`。如果改动影响多个范围，可以省略或选择最主要的。
    -   **`subject`**: 对 PR 所做更改的简洁、祈使句描述（例如 "add" 而不是 "added" 或 "adds"），首字母小写，末尾不加句号。
    -   **示例 PR 标题**:
        ```
        feat(agent): implement information dictionary in get_action method
        fix(training): correct learning rate scheduler logic for AIL cycle
        docs(readme): add detailed project setup and run instructions
        ```
-   **PR 描述 (Description/Body)**:
    -   清晰、详细地描述您所做的更改。
    -   解释这些更改的动机、解决的问题或实现的功能。
    -   如果适用，提供如何测试这些变更的步骤。
    -   如果您的 PR 解决了某个 GitHub Issue，请在描述中链接该 Issue (例如 `Closes #issue_number` 或 `Fixes #issue_number`)。
    -   (可选) 包含截图、设计决策的简要说明等。

#### e. 测试

-   我们鼓励为新功能或重要的 Bug 修复编写单元测试和集成测试。
-   测试的重点应放在验证核心功能的正确性、主要用户场景以及模块间的关键交互上。请不要针对边界条件编写单元测试，我们应该假设上层代码能够正确地调用，过多的单元测试对于review而言也是巨大的负担。
-   对于 Bug 修复，请编写一个能够暴露该 Bug 的测试（如果之前没有）。
-   在提交 Pull Request 之前，请确保所有测试都已通过。
-   单元测试的代码应该放在 `tests/` 文件夹下面， `tests/` 目录结构应该与 `odpc/` 的目录结构保持一致，例如，对 `odpc/data/utils.py` 的测试应位于 `tests/data/test_utils.py`。对于集成测试，也请尽量根据代码结构选择合适的文件位置。
- **测试硬件依赖**: 本项目中的部分测试可能需要特定的硬件环境，例如 NVIDIA GPU 和 CUDA 工具包。
    -   **标记 GPU 测试**: 需要 GPU 的测试用例会使用 `@pytest.mark.gpu` 进行标记。
    -   **自动跳过**: 如果在没有 CUDA 可用环境的设备上运行测试套件，这些 GPU 特定的测试会被自动跳过。
    -   **本地运行 GPU 测试**: 如果您有可用的 GPU 环境，可以通过以下命令专门运行 GPU 测试：
        ```bash
        pytest -m gpu
        ```
    -   **CI 环境**: 我们的 CI 环境目前可能不包含 GPU 资源。因此，GPU 特定代码的全面验证依赖于贡献者在本地 GPU 环境下的测试。请在提交涉及 GPU 计算的更改前，确保在本地 GPU 环境下通过了相关测试。

#### f. 文档

-   如果您的代码更改引入了新的用户可见功能、修改了现有API或核心行为，请务必更新相关的文档。这可能包括：
    -   `README.md`
    -   `ARCHITECTURE.md`
    -   代码中的文档字符串 (Docstrings)
    -   (如果适用) `CHANGELOG.md` 的 `[Unreleased]` 部分。

#### g. Pull Request (PR) 流程与合并

1.  当您的特性分支准备好后，请向项目的 `main` 分支发起一个 Pull Request。
2.  **PR 信息 (标题与描述)**:
    -   请确保您的 Pull Request **标题和描述**都清晰、完整，并严格遵循 **`c.ii. Pull Request (PR) 的标题与描述`** 中定义的要求。
    -   高质量的 PR 信息对于代码审查、历史追溯以及后续 Changelog 的生成至关重要。
3.  确保您的 PR 只包含与目标功能或修复相关的提交。避免混入不相关的更改。
4.  项目维护者或其他贡献者将对您的 PR 进行审查。请积极参与讨论并根据反馈进行修改。
5.  **合并到主分支**:
    -   一旦 PR 通过审查并满足所有要求，它将被合并到 `main` 分支。
    -   项目维护者通常会使用 **"Squash and Merge"** 策略。这意味着您在特性分支上的多个提交将被“压扁”为一个单独的提交记录在 `main` 分支上。
    -   **这个最终合并到 `main` 分支的提交，其 message 将基于您的 PR 标题（作为 subject）和 PR 描述（或其摘要，作为 body）生成。** 因此，再次强调 PR 标题和描述的规范性至关重要，因为它们直接影响主干历史的清晰度和 Changelog 的质量。

## 行为准则

我们致力于为所有贡献者和用户提供一个友好、安全和热情的环境。请在所有互动中保持专业和尊重。具体行为准则请参考 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) (如果未来添加此文件)。目前，请遵循普遍接受的开源社区行为规范。

---

感谢您的贡献！