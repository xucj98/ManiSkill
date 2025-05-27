# Related Works

1. 从人类示范中学习
  - 1.1 开环控制：OKAMI, FUNCTO
  - 1.2 需要机器人本体或者仿真微调的方法：WHIRL，IGOR，XSkill，LAPA，Im2Flow2Act, PoCo
  - 1.3 依赖中间表示的方法
    - 2D/3D光流：Im2Flow2Act, ToolFlowNet, Track2Act
    - 关键点：FUNCTO
    - 6D物体位姿：SPOT，FreePose，Tool-as-Interface
    - Wrist：ZeroMimic, WHIRL
    - Latent action：IGOR，LAPA，XSkill，MimicPlay, RHyME
  - 1.4 使用图像编辑的方法：Phantom, WHIRL
2. 6D物体位姿估计及其在从人类示范中学习中的作用
  - 2.1 6D物体位姿估计的挑战
    - 姿态估计精度问题、模型网格（mesh）需求和实时估计的挑战。
    - 相机到机器人标定的需求、对物体几何形状和遮挡的敏感性。
  - 2.2 使用6D位姿估计进行控制的方法
    - 综述依赖6D物体位姿估计进行控制的相关方法（如SPOT，FreePose，Tool-as-Interface等）。
    - 这些方法在复杂环境中存在实时姿态估计精度的不足。
  - 2.3 与DeltaCtrl的关系
    - DeltaCtrl通过直接从RGBD视频预测位姿变化（delta transformations），避免了对6D物体位姿估计的依赖。
    - 使用相对运动预测（delta T）而不是绝对位姿估计的优势。
3. DeltaCtrl中使用的现有模块
  - 虽然DeltaCtrl的核心贡献在于其独特的从人类示范中学习位姿变化的方法，但我们确实依赖几个现有模块来实现高效和精确的结果。这些模块的选择是基于它们的性能、可扩展性和能够顺利集成到我们系统中的能力。下面简要讨论我们使用的关键模块：
    - 6D物体位姿估计：从RGBD视频中提取物体位姿的方法。例如，[参考文献XYZ] 提供了高精度的位姿估计，而其他方法在动态环境中的表现较差。
    - 工具抓取：用于工具操作和抓取的策略，例如**[参考文献XYZ]** 中基于可供性（affordance）的方法。
    - 视觉-运动策略学习：使用Diffusion Policy、cVAE和**VLA（视觉-语言-行动）**模型来从视觉输入中预测工具的运动和动作轨迹。由于这些方法能够有效处理时间序列并具备较好的可扩展性，因此我们选择了它们。