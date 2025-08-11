"""
运行主动模仿学习 (A-IL) 循环的主入口点。

该脚本根据 ARCHITECTURE.md 中的描述，编排整个 A-IL 工作流程。
它遵循自顶向下的设计，为 A-IL 循环的每个阶段都提供了清晰的函数。
初期，这些函数作为具有明确接口（输入和输出）的占位符，后续将逐一实现。
"""

import argparse
import logging
import pathlib
import multiprocessing as mp
from typing import Dict, Any, List, Optional

import h5py
import numpy as np
from omegaconf import DictConfig, OmegaConf

from odpc.data.demo.generation import run_generation_workflow
from odpc.data.demo.generation_from_state import run_generation_workflow as generation_from_state
from odpc.training.train import train
from odpc.evaluation.rollout import rollout
from odpc.evaluation.rollout_analysis import analyze_rollout

from odpc.utils.utils import load_config_with_defaults, parse_config_expr


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


def generate_expert_demos(
    config: DictConfig,
    output_dir: pathlib.Path,
    initial_states_path: Optional[pathlib.Path] = None,
) -> pathlib.Path:
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
        demo_config.save_dir = str(output_dir)

        generated_path = run_generation_workflow(cfg=demo_config)
       
    else:
        logging.info(f"阶段 5: 正在从高价值状态 {initial_states_path} 生成靶向性演示...")

        analysis_dir = initial_states_path.parent

        demo_config = config.demo_from_state
        demo_config.save_dir = str(output_dir)
        demo_config.motion_planning_args.analysis_dir = str(analysis_dir)

        generated_path = generation_from_state(cfg=demo_config)
       
    logging.info(f"演示已保存至: {generated_path}")
    return pathlib.Path(generated_path)


def train_policy(
    config: DictConfig,
    dataset_paths: List[pathlib.Path],
    output_dir: pathlib.Path,
) -> pathlib.Path:
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
    
    train_config = config.train
    train_config.save_dir = str(output_dir)
    train_config.data_paths = [str(p) for p in dataset_paths]

    train(cfg=train_config)

    model_ckpt_path = output_dir / "checkpoints/best.pt"
    
    logging.info(f"训练好的模型已保存至: {model_ckpt_path}")
    return model_ckpt_path


def policy_rollout(
        config: DictConfig,
        output_dir: pathlib.Path,
        policy_ckpt_path: pathlib.Path,
) -> pathlib.Path:
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

    rollout_config = config.rollout
    rollout_config.save_dir = str(output_dir)

    rollout_log_path = rollout(rollout_config, policy_ckpt_path)
    
    logging.info(f"部署结果保存至: {rollout_log_path}")
    return rollout_log_path


def offline_analysis(
    config: DictConfig,
    output_dir: pathlib.Path,
    rollout_log_path: pathlib.Path,
    dataset_paths: List[pathlib.Path],
    ckpt_path: pathlib.Path,
) -> pathlib.Path:
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
    
    analysis_config = config.rollout_analysis
    analysis_config.save_dir = str(output_dir)
    analysis_config.ckpt_path = str(ckpt_path)
    analysis_config.rollout_path = str(rollout_log_path)
    analysis_config.data_paths = [str(p) for p in dataset_paths]

    key_env_states_path = analyze_rollout(config=analysis_config)

    logging.info(f"离线分析结果已保存至: {key_env_states_path}")
    return key_env_states_path


def main(args, cfg):
    """运行 A-IL 循环的主函数。"""
    # --- 设置多进程启动方式 ---
    # `set_start_method` 只能在程序开始时调用一次。
    # 由于数据生成模块依赖 'spawn' 模式以确保 CUDA 安全，
    # 我们必须在整个应用生命周期中统一使用 'spawn' 模式。
    # PyTorch 的 DataLoader 也完全兼容 'spawn' 模式。
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # 如果已经设置，则忽略

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
    policy_ckpt_path = train_policy(cfg, [initial_dataset_path], cycle_0_dir / "train")

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
        rollout_log_path = policy_rollout(
            cfg, cycle_dir / "rollout", policy_ckpt_path
        )

        # 阶段 4: 离线分析
        key_env_states_path = offline_analysis(
            cfg, cycle_dir / "rollout_analysis", rollout_log_path, 
            aggregated_dataset_paths, policy_ckpt_path
        )

        # 阶段 5: 靶向性演示
        targeted_demos_path = generate_expert_demos(
            cfg, cycle_dir, initial_states_path=key_env_states_path
        )

        # 阶段 6: 聚合与再训练
        aggregated_dataset_paths.append(targeted_demos_path) # 直接追加
        policy_ckpt_path = train_policy(cfg, aggregated_dataset_paths, cycle_dir / "train")

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