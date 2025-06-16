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

    cfg.demo_config = OmegaConf.load(cfg.train_dataset.data_path.replace(".h5", ".yaml"))
    cfg.save_dir = f"runs/{cfg.wandb_group}/{cfg.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return args, cfg


def copy_ema_buffers():
    for ema_buffer, source_buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.data.copy_(source_buffer.data)


def save_ckpt():
    if step % cfg.trainer.save_freq == 0:
        os.makedirs(f"{cfg.save_dir}/checkpoints", exist_ok=True)
        ema.copy_to(ema_model.parameters())
        copy_ema_buffers()
        torch.save(
            {
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
            },
            f"{cfg.save_dir}/checkpoints/{step}.pt",
        )


def evaluate_and_save_best():
    if step % cfg.trainer.eval_freq == 0:
        ema.copy_to(ema_model.parameters())
        copy_ema_buffers()

        metrics = evaluator.evaluate(args.num_eval_episodes, cfg.trainer.batch_size)
        for k, v in metrics.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    writer.add_scalar(f"eval/{k}/{sub_k}", sub_v.mean(), step)
                    print(f"eval/{k}/{sub_k}: {sub_v.mean():.6f}")
            else:
                writer.add_scalar(f"eval/{k}", v.mean(), step)
                print(f"eval/{k}: {v.mean():.6f}")


def log_metrics():
    if step % cfg.trainer.log_freq == 0:
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], step
        )
        writer.add_scalar("losses/total_loss", total_loss.item(), step)
            

if __name__ == "__main__":
    args, cfg = get_args()
    
    assert cfg.obs_horizon + cfg.act_horizon - 1 <= cfg.pred_horizon
    assert cfg.obs_horizon >= 1 and cfg.act_horizon >= 1 and cfg.pred_horizon >= 1

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
            entity=cfg.wandb_entity,
            project=cfg.wandb_project_name,
            sync_tensorboard=True,
            config=OmegaConf.to_object(cfg),
            name=cfg.exp_name,
            save_code=True,
            group=cfg.wandb_group,
            tags=["odpc"],
            job_type="train",
        )

    writer = SummaryWriter(cfg.save_dir)
    
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

    model: BasePolicy = instantiate_from_config(cfg.model).to(device)
    
    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())
    lr_scheduler = instantiate_from_config(cfg.lr_scheduler, optimizer=optimizer)
    
    ema: EMAModel = instantiate_from_config(cfg.ema, parameters=model.parameters())
    ema_model: BasePolicy = instantiate_from_config(cfg.model).to(device)
    ema_model.eval()

    # 创建evaluator
    evaluator: Evaluator = instantiate_from_config(cfg.evaluator, model=ema_model)

    # 初始化混合精度训练
    scaler = GradScaler() if cfg.trainer.use_amp else None
    amp_dtype = getattr(torch, cfg.trainer.amp_dtype) if cfg.trainer.use_amp else torch.float32

    model.train()
    pbar = tqdm(total=cfg.trainer.total_iters)

    for step, data_batch in enumerate(train_dataloader):
        data_batch = common.to_tensor(data_batch, device)
            
        optimizer.zero_grad()
        
        # 使用混合精度训练
        if cfg.trainer.use_amp:
            with autocast(dtype=amp_dtype):
                total_loss = model.compute_loss(
                    obs=data_batch["observations"],
                    action=data_batch["actions"],
                )
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss = model.compute_loss(
                obs=data_batch["observations"],
                action=data_batch["actions"],
            )
            total_loss.backward()
            optimizer.step()
            
        lr_scheduler.step()

        ema.step(model.parameters())
        
        evaluate_and_save_best()
        log_metrics()
        save_ckpt()

        pbar.update(1)
        pbar.set_postfix({"loss": total_loss.item()})

    evaluator.close()
    writer.close()
