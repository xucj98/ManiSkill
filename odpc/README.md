# ODPC (Object Delta Pose Control) 项目

## 项目概述

ODPC是一个基于扩散策略(Diffusion Policy)的机器人操作控制框架，专注于从人类视频中学习精细的操作。本项目在ManiSkill仿真环境中实现。

ODPC的核心创新点在于：
1. 从人类RGBD视频中学习高精度的工具使用操作，无需任何机器人演示数据
2. 通过预测物体在相机坐标系下的位姿变化（ΔT），避免了物体坐标系定义导致的歧义
3. 实现了端到端的视觉-运动策略学习，保留了完整的视觉语义线索
4. 支持闭环控制，能够根据实时反馈调整策略

## 技术原理

ODPC通过以下步骤实现物体操作控制：

1. **位姿变化预测**：
   - 使用Diffusion Policy预测物体在相机坐标系下的位姿变化（ΔT）
   - 避免了物体坐标系定义导致的歧义
   - 支持不同相机设备和物体的泛化

2. **坐标变换**：
   - 通过相机标定和机器人运动学模型
   - 将ΔT转换为末端执行器的目标位姿
   - 实现闭环控制

3. **实时反馈**：
   - 支持基于视觉反馈的实时调整
   - 能够处理动态变化的任务需求
   - 提高操作精度和鲁棒性

## 项目结构

```
odpc/
├── paper/                  # 论文
├── configs/                # 配置文件目录
├── envs/                   # 仿真环境
├── models/                 # 模型实现
├── data/                   # 数据处理相关代码
├── utils/                  # 工具函数
├── scripts/                # 运行脚本
└── experiments/            # 实验结果和日志
└── progress/               # 项目进度
```

## 主要功能

1. **演示轨迹生成**
   - 基于运动规划生成专家演示
   - 支持随机化初始状态
   - 支持多进程并行生成

2. **模型训练**
   - 支持多种backbone (ResNet, DINOv2, PointNet++等)
   - 支持语义分割输入

3. **评估与实验**
   - 支持域内/域外评估
   - 误差分析
   - 支持多种评估指标
   - 实验配置版本控制

## 配置系统

项目使用YAML配置文件管理所有实验参数，包括：

- 模型配置（backbone类型、扩散模型参数等）
- 数据集配置（数据路径、预处理方式等）
- 训练配置（学习率、batch size等）
- 实验配置（评估方式、日志记录等）

每个实验都会记录：
- 完整的配置文件
- Git commit ID
- 实验结果和日志
- 模型检查点

## 待优化方向

1. **模型架构优化**
   - 集成DINOv2作为视觉backbone
   - 实现基于点云的DP3模型
   - 探索多模态融合方案

2. **演示数据增强**
   - 实现3D空间随机化初始状态
   - 增加轨迹多样性
   - 添加噪声和扰动

3. **语义信息利用**
   - 将robot部分mask以减少 cross-embodiment gap
   - 输入 object mask，mask as query 实现对多物体场景的学习


## 使用说明

1. **环境配置**
```bash
# 安装依赖
pip install -r requirements.txt
```

2. **生成演示数据**
```bash
python scripts/generate_demos.py --config configs/demo/peg_insertion.yaml
```

3. **训练模型**
```bash
python scripts/train.py --config configs/experiment/train.yaml
```

4. **评估模型**
```bash
python scripts/evaluate.py --config configs/experiment/eval.yaml
```

## 开发计划

1. **第一阶段：基础设施**
   - [x] 基础ODPC实现
   - [ ] 配置系统重构
   - [ ] 实验管理工具

2. **第二阶段：模型优化**
   - [ ] DINOv2集成
   - [ ] DP3实现

3. **第三阶段：数据增强**
   - [ ] 3D随机化实现
   - [ ] 轨迹多样性增强
   - [ ] 数据增强pipeline

4. **第四阶段：消融实验**
   - [ ] OOD评估
   - [ ] 误差分析
   - [ ] 更多的环境测试
    

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。在提交PR时，请确保：

1. 代码符合项目的代码规范
2. 添加必要的测试
3. 更新相关文档
4. 提供详细的改动说明

## 许可证

本项目采用MIT许可证。详见LICENSE文件。 