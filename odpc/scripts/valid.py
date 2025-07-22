"""
Usage:

1. 评估一个模型: 使用模型训练的config文件
    使用模型训练的config文件, 其中 cfg.evaluator.env_configs.ind.env._target_ = odpc.envs.factory.make_eval_envs.make_eval_envs
    eg. configs/exps/PegIns_EeDeltaPose.yaml

    python scripts/valid.py --config path_to_config.yaml --ckpt-path path_to_checkpoint.pt

2. 评估多个模型, 这些模型在同一个目录下, 根据训练的step命名
    |-- path_to_ckpt_dir/
        |-- 100000.pt
        |-- 200000.pt
        |-- ...
    
    python scripts/valid.py --config path_to_config.yaml --ckpt-path path_to_ckpt_dir

3. 进行rollout, 并记录策略与环境交互的结果
    使用rollout的config文件, 其中 cfg.evaluator.env_configs.ind.env._target_ = odpc.envs.factory.make_eval_envs.make_rollout_envs
    eg. configs/ail/rollout/PegIns_EeDeltaPose.yaml

    python scripts/valid.py --config path_to_config.yaml --ckpt-path path_to_checkpoint.pt
"""

import os
import random
from datetime import datetime
import argparse
from omegaconf import OmegaConf

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

import odpc.envs
from odpc.data.data_conversion import DataConversion
from odpc.models.policy import DiffusionUnetImagePolicy
from odpc.utils.utils import instantiate_from_config, load_config_with_defaults
from odpc.evaluation.evaluate import Evaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/odpc/base.yaml")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    args, unknown = parser.parse_known_args()

    cfg = load_config_with_defaults(args.config)
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(cfg, cli)

    cfg.save_dir = f"runs/{cfg.wandb_group}/{cfg.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cfg.ckpt_path = args.ckpt_path
 
    print(OmegaConf.to_yaml(cfg, resolve=True))
    
    return args, cfg


if __name__ == "__main__":
    args, cfg = get_args()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(cfg.save_dir)
    OmegaConf.save(cfg, f"{cfg.save_dir}/config.yaml", resolve=True)
    print(f"save_dir: {os.path.abspath(cfg.save_dir)}")

    if cfg.track:
        import wandb
        wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            sync_tensorboard=True,
            config=OmegaConf.to_object(cfg),
            name=cfg.exp_name,
            save_code=True,
            group=cfg.wandb_group,
            tags=["odpc"],
            job_type="eval",
        )
    
    writer = SummaryWriter(cfg.save_dir)
    
    model: DiffusionUnetImagePolicy = instantiate_from_config(cfg.model).to(device)

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

    # 创建evaluator
    evaluator: Evaluator = instantiate_from_config(cfg.evaluator, model=model)

    for step, ckpt_path in ckpt_paths.items():
        ckpt = torch.load(ckpt_path)
        if args.use_ema:
            model.load_state_dict(ckpt["ema_model"])
        else:
            model.load_state_dict(ckpt["model"])

        # 进行评估
        metrics = evaluator.evaluate(args.batch_size)

        # 记录评估结果
        for k, v in metrics.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    writer.add_scalar(f"eval/{k}/{sub_k}", sub_v.mean(), step)
                    print(f"eval/{k}/{sub_k}: {sub_v.mean():.6f}")
            else:
                writer.add_scalar(f"eval/{k}", v.mean(), step)
                print(f"eval/{k}: {v.mean():.6f}")

    evaluator.close()
    writer.close()
