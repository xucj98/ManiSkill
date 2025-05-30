import os
import random
import time
import argparse
from omegaconf import OmegaConf
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import tyro

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils import common
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.make_env import make_eval_envs

import odpc.envs
from odpc.data.data_conversion import DataConversion
from odpc.data.odpc_dataset import ODPCDataset
from odpc.models.odpc import ODPCModel
from odpc.models.agent import ODPCAgent
from odpc.utils.utils import instantiate_from_config
from odpc.utils.evaluate import evaluate_on_env


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/odpc/base.yaml")
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--num-eval-episodes", type=int, default=100)
    parser.add_argument("--outputs-dir", type=str, default='outputs')
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--render-pred", action="store_true", default=False)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(cfg, cli)
 
    return args, cfg


if __name__ == "__main__":
    args, cfg = get_args()

    
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
        video_dir=f"{args.outputs_dir}/video" if args.render else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )

    tmp_env = gym.make(cfg.valid_env.env_id, **env_kwargs)
    origin_obs_space = tmp_env.observation_space
    tmp_env.close()

    if cfg.track:
        import wandb
        wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            config=OmegaConf.to_object(cfg),
            name=cfg.exp_name,
            save_code=True,
            group="ODPC",
            tags=["odpc"],
            job_type="eval",
        )
    
    writer = SummaryWriter(f"runs/{cfg.exp_name}")
    
    data_conversion = DataConversion(
        control_mode=cfg.control_mode,
    )

    model: ODPCModel = instantiate_from_config(cfg.model).to(device)

    agent: ODPCAgent = instantiate_from_config(
        cfg.agent, model=model, dc=data_conversion, origin_obs_space=origin_obs_space,
        video_dir=f"{args.outputs_dir}/pred" if args.render_pred else None,
    )

    if os.path.isdir(args.ckpt_path):
        files = [x for x in os.listdir(args.ckpt_path) if x.endswith(".pt")]
        steps = []
        for file in files:
            try:
                step = int(file.split(".")[0])
                steps.append(step)
            except ValueError:
                pass
        steps.sort()
        ckpt_paths = {step: os.path.join(args.ckpt_path, f"{step}.pt") for step in steps}
    else:
        ckpt_paths = {0: args.ckpt_path}

    for step, ckpt_path in ckpt_paths.items():
        ckpt = torch.load(ckpt_path)
        if args.use_ema:
            model.load_state_dict(ckpt["ema_model"])
        else:
            model.load_state_dict(ckpt["model"])

        eval_metrics = evaluate_on_env(
            args.num_eval_episodes, agent, envs, device, cfg.valid_env.sim_backend)

        for k, v in eval_metrics.items():
            writer.add_scalar(f"eval/{k}", v.mean(), step)
            print(f"eval/{k}: {v.mean():.4f}")

    envs.close()
    writer.close()
