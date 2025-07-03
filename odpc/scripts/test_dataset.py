import os
import random
import argparse
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.cuda.amp import autocast, GradScaler

from diffusers.training_utils import EMAModel

import gymnasium as gym
from mani_skill.utils import common
from diffusion_policy.utils import IterationBasedBatchSampler, worker_init_fn

import odpc.envs
from odpc.utils import utils
from odpc.utils.utils import instantiate_from_config
from odpc.data.data_conversion import DataConversion
from odpc.models.policy import BasePolicy
from odpc.evaluation.evaluate import Evaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/odpc/base.yaml")
    parser.add_argument("--num-eval-episodes", type=int, default=100)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(cfg, cli)

    # cfg.demo_config = OmegaConf.load(
    #     cfg.train_dataset.data_path.replace(".compressed.", ".").replace(".h5", ".yaml"))
    # cfg.save_dir = f"runs/{cfg.wandb_group}/{cfg.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # print(OmegaConf.to_yaml(cfg, resolve=True))

    return args, cfg

if __name__ == "__main__":
    args, cfg = get_args()
    
    assert cfg.obs_horizon + cfg.act_horizon - 1 <= cfg.pred_horizon
    assert cfg.obs_horizon >= 1 and cfg.act_horizon >= 1 and cfg.pred_horizon >= 1

    dataset = instantiate_from_config(cfg.train_dataset)
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=cfg.trainer.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, cfg.trainer.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.trainer.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=cfg.seed),
        persistent_workers=(cfg.trainer.num_dataload_workers > 0),
        pin_memory=True,
    )

    for batch in train_dataloader:
        print(utils.get_data_shape(batch))
        break
