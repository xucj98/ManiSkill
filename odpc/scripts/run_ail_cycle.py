"""
运行主动模仿学习 (A-IL) 循环的主入口点。

该脚本根据 ARCHITECTURE.md 中的描述，编排整个 A-IL 工作流程。
它遵循自顶向下的设计，为 A-IL 循环的每个阶段都提供了清晰的函数。
初期，这些函数作为具有明确接口（输入和输出）的占位符，后续将逐一实现。
"""

import argparse
import logging
import pathlib
from typing import Dict, Any, List, Optional

import h5py
import numpy as np
import pickle
from omegaconf import DictConfig, OmegaConf

from odpc.data.demo.generation import run_generation_workflow
from odpc.utils.utils import load_config_with_defaults, parse_config_expr
# 在注释中定义核心数据结构的格式，以保持清晰。
# 后续这些将被正式定义为类或 TypedDict。

# TrajectoryLog (轨迹日志): 一个字典或 HDF5 文件，包含:
# - "trajectories": 一个轨迹列表。
# - 每个轨迹是一个步骤列表。
# - 每个步骤是一个字典: {
#     "observation": np.ndarray,
#     "action": np.ndarray,
#     "reward": float,
#     "done": bool,
#     "info": {
#         "model_uncertainty": float, # 用于离线分析的关键数据
#         ... # 其他有用的调试信息
#     }
# }

# HighValueStateList (高价值状态列表): 一个状态列表，保存为 .pkl 或 .json 文件。
# 每个状态是一个完整的、可序列化的表示，可用于将环境重置到该特定点。


def _create_dummy_h5_traj(h5_path: pathlib.Path, num_traj: int = 1, traj_len: int = 2):
    """创建一个虚拟的、结构合法的HDF5轨迹文件。"""
    with h5py.File(h5_path, "w") as f:
        for i in range(num_traj):
            traj_group = f.create_group(f"traj_{i}")
            # (T, action_dim)
            traj_group.create_dataset("actions", data=np.random.rand(traj_len, 7).astype(np.float32))
            obs_group = traj_group.create_group("obs")
            extra_group = obs_group.create_group("extra")
            # (T+1, 7)
            extra_group.create_dataset("peg_pose", data=np.random.rand(traj_len + 1, 7).astype(np.float32))
            sensor_group = obs_group.create_group("sensor_data")
            camera_group = sensor_group.create_group("base_camera")
            # (T+1, H, W, C)
            camera_group.create_dataset("rgb", data=np.random.randint(0, 256, (traj_len + 1, 128, 128, 3), dtype=np.uint8))


def setup_logging(cycle_dir: pathlib.Path):
    """为 A-IL 循环设置日志记录。"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(cycle_dir / "ail_cycle.log"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"日志记录已初始化。日志文件: {cycle_dir / 'ail_cycle.log'}")


def generate_expert_demos(config: DictConfig, output_dir: pathlib.Path, initial_states_path: Optional[pathlib.Path] = None) -> pathlib.Path:
    """
    阶段 1 & 5: 生成专家演示。
    根据 `initial_states_path` 是否为 None，决定是生成初始演示还是靶向性演示。

    Args:
        config: 实验配置。
        output_dir: 演示数据集的保存目录。
        initial_states_path: 可选。高价值状态列表的路径。如果提供，则从这些状态开始生成演示。
                                 如果为 None，则生成初始的随机演示。

    Returns:
        生成的演示数据集的路径 (例如, HDF5 文件)。
    """
    if initial_states_path is None:
        logging.info("阶段 1: 正在生成初始专家演示...")
        demo_config = config.demo
        # 注意：这里的 jpg_quality 是硬编码的，未来可以加入到配置中
        generated_path_str = run_generation_workflow(cfg=demo_config, record_dir=str(output_dir), jpg_quality=90)
        
        # 直接使用生成的文件路径，不进行重命名
        demo_path = pathlib.Path(generated_path_str)

    else:
        logging.info(f"阶段 5: 正在从高价值状态 {initial_states_path} 生成靶向性演示...")
        # TODO: 实现加载高价值状态并辅助专家采集的逻辑。
        #       这需要一个新的函数，它能接收 initial_states_path 并将其内容传递给数据生成流程。
        #       暂时创建一个虚拟文件。
        demo_name = "targeted_demos.h5"
        demo_path = output_dir / demo_name
        _create_dummy_h5_traj(demo_path, num_traj=config.ail_loop.targeted_demos_per_cycle)

    logging.info(f"演示已保存至: {demo_path}")
    return demo_path


def train_policy(config: DictConfig, dataset_paths: List[pathlib.Path], output_dir: pathlib.Path) -> pathlib.Path:
    """
    阶段 2 & 6: 在给定数据集上训练策略模型。
    对应 `scripts/train_policy.py` 并调用 `odpc.training.trainer`。

    Args:
        config: 实验配置。
        dataset_paths: 训练数据集的路径列表。
        output_dir: 用于保存训练好的模型检查点的目录。

    Returns:
        训练好的模型检查点的路径。
    """
    logging.info(f"正在使用数据集进行策略训练: {dataset_paths}")
    
    model_ckpt_path = output_dir / "policy_best.ckpt"

    # TODO: 通过调用 `odpc.training.trainer` 来实现此逻辑。
    # `odpc.data.odpc_dataset.ImitationDataset` 需要支持加载一个包含多个 HDF5 文件路径的列表，
    # 并能够将它们聚合为一个统一的数据集进行训练。
    model_ckpt_path.touch()  # 创建一个虚拟文件
    
    logging.info(f"训练好的模型已保存至: {model_ckpt_path}")
    return model_ckpt_path


def policy_rollout(config: DictConfig, cycle_dir: pathlib.Path, policy_ckpt_path: pathlib.Path) -> pathlib.Path:
    """
    阶段 3: 在环境中部署当前策略以收集轨迹。

    Args:
        config: 实验配置。
        cycle_dir: 当前轮次产物的存放目录。
        policy_ckpt_path: 需要部署的策略模型的路径。

    Returns:
        收集到的轨迹日志文件的路径 (TrajectoryLog 格式)。
    """
    logging.info(f"阶段 3: 正在部署策略: {policy_ckpt_path}")
    rollout_log_path = cycle_dir / "rollout_log.h5"

    # TODO: 实现加载策略与智能体。
    # TODO: 通过一个环境工厂函数来创建并包装环境，确保环境被 `AILRecordEpisode` 包装器正确应用，
    # 并配置好轨迹保存路径和文件名。
    # TODO: 调用 `odpc.odpc.evaluation.evaluate.evaluate_on_env` 函数来执行策略部署。
    # TODO: 策略部署的起始状态将采用随机初始化。
    rollout_log_path.touch()

    logging.info(f"部署日志已保存至: {rollout_log_path}")
    return rollout_log_path


def offline_analysis(config: DictConfig, cycle_dir: pathlib.Path, rollout_log_path: pathlib.Path) -> pathlib.Path:
    """
    阶段 4: 分析部署日志，以识别用于下一轮的高价值状态。

    Args:
        config: 实验配置。
        cycle_dir: 当前轮次产物的存放目录。
        rollout_log_path: 策略部署阶段产出的轨迹日志文件路径。

    Returns:
        识别出的高价值状态列表的路径 (HighValueStateList 格式)。
    """
    logging.info(f"阶段 4: 正在对日志进行离线分析: {rollout_log_path}")
    high_value_states_path = cycle_dir / "high_value_states.pkl"

    # TODO: 实现 A-IL 的核心逻辑:
    # 1. 筛选失败/次优轨迹。
    # 2. 根据不确定性/影响力对状态进行排序。
    # 3. 采样一个多样化的高价值状态集合。
    high_value_states_path.touch()

    logging.info(f"高价值状态已保存至: {high_value_states_path}")
    return high_value_states_path


def main(args, cfg):
    """运行 A-IL 循环的主函数。"""
    # --- 设置基础目录和日志 ---
    cfg.output_dir = parse_config_expr(cfg.output_dir)
    base_dir = pathlib.Path(cfg.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(base_dir)

    # --- 第 0 轮: 初始化 ---
    cycle_0_dir = base_dir / "cycle_0"
    cycle_0_dir.mkdir(exist_ok=True)

    logging.info("=" * 50)
    logging.info("启动 A-IL 第 0 轮: 初始化")
    logging.info("=" * 50)

    # 阶段 1: 生成初始专家演示
    initial_dataset_path = generate_expert_demos(cfg, cycle_0_dir, initial_states_path=None)
    
    # 阶段 2: 初始模型训练
    policy_ckpt_path = train_policy(cfg, [initial_dataset_path], cycle_0_dir)

    # 聚合后的数据集路径列表
    aggregated_dataset_paths = [initial_dataset_path]

    # --- A-IL 主循环 ---
    for i in range(1, cfg.ail_loop.num_cycles + 1):
        cycle_dir = base_dir / f"cycle_{i}"
        cycle_dir.mkdir(exist_ok=True)

        logging.info("=" * 50)
        logging.info(f"启动 A-IL 第 {i} 轮")
        logging.info("=" * 50)

        # 阶段 3: 策略部署
        rollout_log_path = policy_rollout(cfg, cycle_dir, policy_ckpt_path)

        # 阶段 4: 离线分析
        high_value_states_path = offline_analysis(cfg, cycle_dir, rollout_log_path)

        # 阶段 5: 靶向性演示
        targeted_demos_path = generate_expert_demos(cfg, cycle_dir, initial_states_path=high_value_states_path)

        # 阶段 6: 聚合与再训练
        aggregated_dataset_paths.append(targeted_demos_path) # 直接追加
        policy_ckpt_path = train_policy(cfg, aggregated_dataset_paths, cycle_dir)

        logging.info(f"已完成 A-IL 第 {i} 轮。当前策略: {policy_ckpt_path}")

    logging.info("=" * 50)
    logging.info("A-IL 流程已结束。")
    logging.info("=" * 50)


def get_args():
    parser = argparse.ArgumentParser(description="运行主动模仿学习 (A-IL) 循环。")
    parser.add_argument("--config", type=str, default="configs/ail/ail_smoke_test.yaml")
    args, unknown = parser.parse_known_args()
    
    cfg = load_config_with_defaults(args.config)
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(cfg, cli)

    return args, cfg

if __name__ == "__main__":
    args, cfg = get_args()
    main(args, cfg)