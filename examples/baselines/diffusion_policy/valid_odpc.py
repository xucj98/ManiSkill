ALGO_NAME = "BC_Diffusion_rgbd_UNet"

import os
import random
import time
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
from diffusion_policy.utils import (IterationBasedBatchSampler,
                                    build_state_obs_extractor, convert_obs,
                                    worker_init_fn)
from diffusion_policy.data_converison import DataConversion
from diffusion_policy.odpc_dataset import ODPCDataset
from diffusion_policy.agent import ODPCAgent, ODPCAgentWrapper
from diffusion_policy.evaluate import evaluate_odpc, evaluate_odpc_on_dataset


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

    env_id: str = "PegInsertionSide-v2"
    """the id of the environment"""
    val_demo_path_ind: str = None
    """the path of in-domain validation dataset"""
    val_demo_path_ood: str = None
    """the path of out-domain validation dataset"""
    robot_ind: str = "panda_peg_insertion"
    """in-domain robot"""
    robot_ood: str = "xarm6_robotiq"
    """out-domain robot"""

    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 400_000
    """total timesteps of the experiment"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""

    # Diffusion Policy specific arguments
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    obs_horizon: int = 1  # Seems not very important in ManiSkill, 1, 2, 4 work well
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
    obs_mode: str = "state_dict+rgb+depth"
    """The observation mode to use for the environment, which dictates what visual inputs to pass to the model. Can be "rgb", "depth", or "rgb+depth"."""
    max_episode_steps: Optional[int] = 300
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 40_000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 8
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_ee_delta_pose"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""
    use_ema: bool = True
    """Whether use ema weight."""

    # Observation process arguments
    depth_clamp: int = 3000
    """Clamp depth value to [0, threshold], depth value beyond threshold will be set to 0."""
    used_cameras: str = "0"
    """Camera used as DP input. Can be camera ids, such as "0", "0,1,2,"."""
    use_state: bool = False
    """Whether to use state as DP input."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None



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
        robot_uids=args.robot_ind,
    )
    assert args.max_episode_steps != None, "max_episode_steps must be specified as imitation learning algorithms task solve speed is dependent on the data you train on"
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    envs_ind = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos/ind" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )
    tmp_env = gym.make(args.env_id, **env_kwargs)
    obs_space_ind = tmp_env.observation_space
    tmp_env.close()

    env_kwargs['robot_uids'] = args.robot_ood
    envs_ood = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos/ood" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )
    tmp_env = gym.make(args.env_id, **env_kwargs)
    obs_space_ood = tmp_env.observation_space
    tmp_env.close()

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

    val_dataset_ind = val_dataset_ood = None
    if args.val_demo_path_ind is not None:
        val_dataset_ind = ODPCDataset(
            data_path=args.val_demo_path_ind,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            num_traj=100,
        )
    if args.val_demo_path_ood is not None:
        val_dataset_ood = ODPCDataset(
            data_path=args.val_demo_path_ood,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            num_traj=100,
        )

    data_conversion = DataConversion(
        control_mode=args.control_mode,
    )

    agent = ODPCAgent(envs_ind, args, data_conversion.pred_dim).to(device)
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

    agent_ind = ODPCAgentWrapper(agent, envs_ind, data_conversion, obs_space_ind, args.robot_ind,
                                 f"runs/{run_name}/videos/ind" if args.capture_video else None)
    agent_ood = ODPCAgentWrapper(agent, envs_ood, data_conversion, obs_space_ood, args.robot_ood,
                                 f"runs/{run_name}/videos/ood" if args.capture_video else None)

    best_eval_metrics = defaultdict(float)

    for step, ckpt_path in ckpt_paths.items():
        ckpt = torch.load(ckpt_path)
        if args.use_ema:
            agent.load_state_dict(ckpt["ema_agent"])
        else:
            agent.load_state_dict(ckpt["agent"])

        eval_metrics = evaluate_odpc(
            args.num_eval_episodes, agent_ind, envs_ind, device, args.sim_backend
        )

        print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
        for k in eval_metrics.keys():
            eval_metrics[k] = np.mean(eval_metrics[k])
            writer.add_scalar(f"eval/{k}", eval_metrics[k], step)
            print(f"{k}: {eval_metrics[k]:.4f}")

        # if val_dataset_ind is not None:
        #     other_metrics = evaluate_odpc_on_dataset(
        #         val_dataset_ind,
        #         agent,
        #         data_conversion,
        #         args,
        #         device,
        #         # video_dir=f"runs/{run_name}/videos/ind" if args.capture_video else None,
        #     )
        #     for k, v in other_metrics.items():
        #         writer.add_scalar(f"eval/{k}", v, step)
        #         print(f"{k}: {v:.4f}")

        eval_ood_metrics = evaluate_odpc(
            args.num_eval_episodes, agent_ood, envs_ood, device, args.sim_backend
        )
        for k in eval_ood_metrics.keys():
            eval_ood_metrics[k] = np.mean(eval_ood_metrics[k])
            writer.add_scalar(f"eval-ood/{k}", eval_ood_metrics[k], step)
            print(f"{k}: {eval_ood_metrics[k]:.4f}")


    envs_ood.close()
    envs_ind.close()
    writer.close()