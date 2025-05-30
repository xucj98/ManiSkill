import os
import time
import random
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import RandomSampler, BatchSampler

from diffusers.training_utils import EMAModel

import gymnasium as gym
from mani_skill.utils import common
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.utils import IterationBasedBatchSampler, worker_init_fn

import odpc.envs
from odpc.utils.utils import instantiate_from_config
from odpc.data.data_conversion import DataConversion
from odpc.models.odpc import ODPCModel
from odpc.utils.evaluate import Evaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/odpc/base.yaml")
    parser.add_argument("--exp-name", type=str, default="odpc-train")
    parser.add_argument("--num-eval-episodes", type=int, default=100)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(cfg, cli)

    cfg.exp_name = args.exp_name
 
    return args, cfg


def copy_ema_buffers():
    for ema_buffer, source_buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.data.copy_(source_buffer.data)


def save_ckpt():
    if step % cfg.trainer.save_freq == 0:
        os.makedirs(f"runs/{cfg.exp_name}/checkpoints", exist_ok=True)
        ema.copy_to(ema_model.parameters())
        copy_ema_buffers()
        torch.save(
            {
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
            },
            f"runs/{cfg.exp_name}/checkpoints/{step}.pt",
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

    if cfg.track:
        import wandb
        wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            sync_tensorboard=True,
            config=OmegaConf.to_object(cfg),
            name=cfg.exp_name,
            save_code=True,
            group="ODPC",
            tags=["odpc"],
            job_type="train",
        )

    writer = SummaryWriter(f"runs/{cfg.exp_name}")
    
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

    data_conversion = instantiate_from_config(cfg.data_conversion)

    model: ODPCModel = instantiate_from_config(cfg.model).to(device)
    
    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())
    lr_scheduler = instantiate_from_config(cfg.lr_scheduler, optimizer=optimizer)
    
    ema = EMAModel(parameters=model.parameters(), power=0.75)  
    ema_model: ODPCModel = instantiate_from_config(cfg.model).to(device)
    ema_model.eval()

    # 创建evaluator
    evaluator: Evaluator = instantiate_from_config(cfg.evaluator, model=model, dc=data_conversion)

    model.train()
    pbar = tqdm(total=cfg.trainer.total_iters)

    for step, data_batch in enumerate(train_dataloader):
        data_batch = common.to_tensor(data_batch, device)

        pred = data_conversion.raw_to_pred(
            poses_obj=data_batch['poses_peg'],
            poses_camera_world=data_batch['poses_cam0_world'],
        )

        total_loss = model.compute_loss(
            obs_seq=data_batch["observations"],
            action_seq=pred,
        )

        optimizer.zero_grad()
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
