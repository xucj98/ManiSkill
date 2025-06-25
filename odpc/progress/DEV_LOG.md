# 开发日志 (DEV_LOG.md)

本文档记录项目开发过程中的重要决策、遇到的问题、解决方案、实验结果和关键思考。
按日期倒序记录，最新的日志在最上方。

---

## 2025-06-25

### “精度诅咒”实验结果量化和可视化 

*   **状态**:
    *   完成 `Reporter` 第二部分，即各个 目标metric 下，num_data w.r.t. precision 的曲线。
    *   在 `Reporter` 中抽象出 `_report_single_group` 和 `_report_groups` 两个函数，可以被两个部分共用，负责绘图和数据分析


## 2025-06-24

### “精度诅咒”实验结果量化和可视化 

*   **动机**: 实现“精度诅咒”分析脚本，保证通用性和可配置性，支持未来不同任务/数据源。
*   **核心架构**:
    *   `Analyzer`: 顶层协调，`scripts/analyze_precision_curse.py` 驱动。
    *   `Reader`: 数据源交互 (当前 `WandbReader`)。
        *   支持灵活的 "OR of ANDs" 筛选器配置。
    *   `Processor`: `data_schema` 驱动，将原始数据转为标准 `pd.DataFrame`。
        *   从原始数据中提取出分析所需要的三个数据：`num_data`, `metric`, `precision`
        *   `transform` 统一处理字段数据，可以支持如 log(x), log(1-x) 等逐点操作 或 max(x) 等聚合操作
    *   `Reporter`: 基于 `Processor` 输出的 DataFrame 进行绘图/分析。
*   **配置**: OmegaConf + YAML，通过 `_target_` 和自定义 `instantiate` 实现组件化。
*   **关键接口**:
    *   `Reader` -> `List[原始运行对象]` -> `Processor`
    *   `Processor` -> `pd.DataFrame` -> `Reporter`
*   **讨论点**:
    *   `Reader`/`Processor` 分离：确认分离设计，以 `Processor` 输出的 DataFrame 为标准中间格式，有利于模块化和未来扩展不同数据源。
    *   WandB 筛选：`WandbReader` 目前客户端筛选，大规模项目下可能需优化。可以看看 `wandb` 是否提供了通过`filter` 读取数据的api接口
*   **状态**: 
    *   完成 `Reader`, `Processor`，`Transforms` 代码。
    *   完成 `Analyzer.run()` 和 `scripts/analyze_precision_curse.py` 框架代码。
    *   完成 `Reporter` 第一部分，即各个精度下， metric w.r.t. num_data 的曲线。
*   **下一步**:  
    *   `Report` 第二部分，即各个 目标metric 下，num_data w.r.t. precision 的曲线。

### demo存储优化
- **背景**: 当前demo的数据集太大了。
- **行动**:
    - 使用 jpeg 压缩 rgb 图片，使用默认85的压缩质量，相比于 gzip 压缩，可以获得约 5.6x 的压缩比
    - 放弃使用 zfp 压缩 depth图片，因为其相比于 gzip 压缩，反而需要 4.2x 的空间


## 2025-06-23

### 创建ODPC项目Cursor开发规则
- **背景**: 为了提高AI助手在ODPC项目开发中的效率和一致性，需要创建一个通用的cursor rule，让AI在每次对话前都能参考项目的关键文档。
- **行动**:
    - 创建了 `.cursor/rules/odpc_development_rule.md` 文件
    - 规则要求AI在每次对话前参考以下文档：
        - `README.md`: 项目整体介绍和核心贡献
        - `ARCHITECTURE.md`: 详细软件架构和模块划分
        - `CONTRIBUTING.md`: 编码规范和提交规范
        - `progress/TODOLIST.md`: 当前任务列表和优先级
        - `progress/DEV_LOG.md`: 开发日志和重要决策
    - 规则包含了核心开发原则、技术架构遵循、开发规范、当前开发重点等内容
    - 更新了 `TODOLIST.md`，将"编写Cursor Rules"标记为已完成
- **价值**:
    - 确保AI助手能够快速理解项目背景和当前状态
    - 提高代码建议的一致性和质量
    - 减少重复解释项目架构和规范的时间
- **下一步**: 在后续开发中验证和优化这个规则的效果

### 初始化开发日志与项目文档体系
- **背景**: 为了更好地管理项目进度、沉淀开发经验并方便与 AI 助手协作，决定开始维护 `DEV_LOG.md` 和 `TODOLIST.md`。
- **行动**:
    - 与 AI 助手共同梳理并最终确定了 `ARCHITECTURE.md`, `CONTRIBUTING.md`, `CHANGELOG.md` (历史版本) 和 `TODOLIST.md` (当前规划) 的内容。
    - 明确了 `TODOLIST.md` 中各主要任务模块的优先级 (P0, P1, P2)。
    - 讨论并确定了 `DEV_LOG.md` 的记录原则和粒度：基于"事件"或"里程碑"，关注"值得记录"的内容，平衡记录负担与价值。
- **下一步**:
    - 开始执行 `TODOLIST.md` 中 P0 优先级的任务，例如"代码文档整理"中的剩余项和"基础设施与效率优化"。
    - 从今天开始，尝试在开发过程中，当遇到重要节点或有值得记录的思考时，更新此 `DEV_LOG.md`。
