import os
import pathlib
from omegaconf import OmegaConf, DictConfig

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

import odpc.envs
from odpc.data.data_conversion import DataConversion
from odpc.models.policy import DiffusionUnetImagePolicy
from odpc.utils.utils import instantiate_from_config, load_config_with_defaults
from odpc.evaluation.evaluate import Evaluator


def rollout(
        cfg: DictConfig,
        ckpt_path: str,
        use_ema: bool = True,
) -> pathlib.Path:
    os.makedirs(cfg.save_dir)
    OmegaConf.save(cfg, f"{cfg.save_dir}/config.yaml", resolve=True)
    print(f"save_dir: {os.path.abspath(cfg.save_dir)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model: DiffusionUnetImagePolicy = instantiate_from_config(cfg.model).to(device)

    evaluator: Evaluator = instantiate_from_config(cfg.evaluator, model=model)
    
    ckpt = torch.load(ckpt_path)
    if use_ema:
        model.load_state_dict(ckpt["ema_model"])
    else:
        model.load_state_dict(ckpt["model"])

    _ = evaluator.evaluate()
    
    evaluator.close()

    return pathlib.Path(cfg.save_dir) / f"{cfg.evaluator.env_configs.ind.env.traj_name}.h5"