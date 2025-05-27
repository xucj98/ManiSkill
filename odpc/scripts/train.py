import os
import time
import argparse
from omegaconf import OmegaConf
import random
import gymnasium as gym
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import RandomSampler, BatchSampler

from diffusers.training_utils import EMAModel

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.utils import IterationBasedBatchSampler, worker_init_fn

from odpc.configs.paths import *
import odpc.envs
from odpc.utils.utils import instantiate_from_config
from odpc.data.data_converison import DataConversion
from odpc.models.odpc import ODPCModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/odpc/base.yaml")
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(cfg, cli)
 
    return args, cfg


if __name__ == "__main__":
    args, cfg = get_args()
    
    run_name = f"{cfg.valid_env.env_id}__{cfg.exp_name}__{cfg.seed}__{int(time.time())}"

    assert cfg.obs_horizon + cfg.act_horizon - 1 <= cfg.pred_horizon
    assert cfg.obs_horizon >= 1 and cfg.act_horizon >= 1 and cfg.pred_horizon >= 1

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_kwargs = OmegaConf.to_object(cfg.valid_env.env_kwargs)
    other_kwargs = dict(obs_horizon=cfg.obs_horizon)
    envs = make_eval_envs(
        cfg.valid_env.env_id,
        cfg.valid_env.num_eval_envs,
        cfg.valid_env.sim_backend,
        env_kwargs,
        other_kwargs,
        wrappers=[FlattenRGBDObservationWrapper],
    )

    tmp_env = gym.make(cfg.valid_env.env_id, **env_kwargs)
    obs_space = tmp_env.observation_space
    tmp_env.close()

    if cfg.track:
        import wandb
        wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            config=OmegaConf.to_object(cfg),
            name=run_name,
            save_code=True,
            group="ODPC",
            tags=["odpc"],
            job_type="train",
        )

    writer = SummaryWriter(f"runs/{run_name}")
    
    cfg.train_dataset.data_path = os.path.join(
        DATASET_ROOT, cfg.valid_env.env_id, 'motionplanning', 
        cfg.train_dataset.data_path)
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
    )

    cfg.valid_dataset.data_path = os.path.join(
        DATASET_ROOT, cfg.valid_env.env_id, 'motionplanning', 
        cfg.valid_dataset.data_path)
    valid_dataset = instantiate_from_config(cfg.valid_dataset)

    data_conversion = DataConversion(
        control_mode=cfg.control_mode,
    )

    odpc_model: ODPCModel = instantiate_from_config(cfg.model).to(device)

    optimizer = instantiate_from_config(cfg.optimizer, params=odpc_model.parameters())
    lr_scheduler = instantiate_from_config(cfg.lr_scheduler, optimizer=optimizer)
    
    ema = EMAModel(parameters=odpc_model.parameters(), power=0.75)
    ema_odpc_model: ODPCModel = instantiate_from_config(cfg.model).to(device)
