ALGO_NAME = "BC_Diffusion_rgbd_UNet"

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.agent import RGBDAgent
from diffusion_policy.evaluate import evaluate, evaluate_on_dataset
from diffusion_policy.make_env import make_eval_envs
from torchvision.models.resnet import resnet18


@dataclass
class Args:
    ckpt_path: str

    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PegInsertionSide-v1"
    """the id of the environment"""

    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""

    # Diffusion Policy specific arguments
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    obs_horizon: int = 2  # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8  # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = (
        16
        # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    )
    diffusion_step_embed_dim: int = 64  # not very important
    unet_dims: List[int] = field(
        default_factory=lambda: [64, 128, 256]
    )  # default setting is about ~4.5M params
    n_groups: int = (
        8  # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are simila
    )

    # Environment/experiment specific arguments
    obs_mode: str = "rgb+depth"
    """The observation mode to use for the environment, which dictates what visual inputs to pass to the model. Can be "rgb", "depth", or "rgb+depth"."""
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""
    use_ema: bool = True
    """Whether use ema weight."""

    # Observation process arguments
    depth_clamp: int = 3000
    """Clamp depth value to [0, threshold], depth value beyond threshold will be set to 0."""
    used_cameras: str = "all"
    """Camera used as DP input. Can be camera ids, such as "0", "0,1,2,"."""
    use_state: bool = True
    """Whether to use state as DP input."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None


def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if k not in d:
            continue
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # create evaluation environment
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
        sensor_configs=dict(shader_pack="default"),
    )
    assert args.max_episode_steps != None, "max_episode_steps must be specified as imitation learning algorithms task solve speed is dependent on the data you train on"
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )

    if args.track:
        import wandb

        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id,
                                      env_horizon=args.max_episode_steps)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="DiffusionPolicy",
            tags=["diffusion_policy"],
            job_type="eval",
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    agent = RGBDAgent(envs, args).to(device)
    if os.path.isdir(args.ckpt_path):
        files = os.listdir(args.ckpt_path)
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

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    for step, ckpt_path in ckpt_paths.items():
        ckpt = torch.load(ckpt_path)
        if args.use_ema:
            agent.load_state_dict(ckpt["ema_agent"])
        else:
            agent.load_state_dict(ckpt["agent"])

        last_tick = time.time()
        eval_metrics = evaluate(
            args.num_eval_episodes, agent, envs, device, args.sim_backend
        )
        # other_metrics = evaluate_on_dataset(val_dataset, agent, args, device)
        # for k, v in other_metrics.items():
        #     eval_metrics[k] = v
        timings["eval"] += time.time() - last_tick

        print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
        for k in eval_metrics.keys():
            eval_metrics[k] = np.mean(eval_metrics[k])
            writer.add_scalar(f"eval/{k}", eval_metrics[k], step)
            print(f"{k}: {eval_metrics[k]:.4f}")

    envs.close()
    writer.close()
