from collections import defaultdict
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from typing import Dict

import gymnasium as gym
from mani_skill.utils import common
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.make_env import make_eval_envs

from odpc.data.data_conversion import pose_multiply, DataConversion
from odpc.models.agent import ODPCAgent
from odpc.models.odpc import ODPCModel
from odpc.utils.utils import instantiate_from_config


def evaluate_on_env(
        n: int, agent: ODPCAgent, eval_envs, device, sim_backend: str, progress_bar: bool = True
)-> Dict[str, np.ndarray]:
    if progress_bar:
        pbar = tqdm(total=n)
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        agent.reset(obs)
        eps_count = 0
        while eps_count < n:
            obs = common.to_tensor(obs, device)
            action_seq = agent.get_action(obs, channel_last=True)
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()
            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                if truncated.any():
                    break
            if truncated.any():
                assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                eps_count += eval_envs.num_envs
                if progress_bar:
                    pbar.update(eval_envs.num_envs)
                obs, info = eval_envs.reset()
                agent.reset(obs)
                
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics

def evaluate_on_dataset(
        model: ODPCModel, dataset, dc, batch_size, device, video_dir=None
)-> Dict[str, np.ndarray]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    square_error = 0.
    n = 0.
    pbar = tqdm(total=len(dataloader))
    for data in dataloader:
        data = common.to_tensor(data, device)

        ground_truth = dc.raw_to_pred(
            poses_obj=data['poses_peg'],
            poses_camera_world=data['poses_cam0_world'],
        )

        observations = data["observations"]
        prediction = model.get_action(observations)

        if video_dir is not None:
            poses_cam_world_cur = data['poses_cam0_world'][..., :1, :]
            poses_peg_cur = data['poses_peg'][..., :1, :]
            poses_cam_obj_cur = pose_multiply(poses_cam_world_cur, poses_peg_cur)
            images_pred = dc.pred_to_visualize(
                rgb=data["observations"]["rgb"][..., :3, :, :],
                pred=prediction,
                poses_cam_obj_cur=poses_cam_obj_cur,
            )[..., ::-1]
            images_gt = dc.pred_to_visualize(
                rgb=data["observations"]["rgb"][..., :3, :, :],
                pred=ground_truth,
                poses_cam_obj_cur=poses_cam_obj_cur,
            )[..., ::-1]
            for i, (img_pred, img_gt) in enumerate(zip(images_pred, images_gt)):
                cv2.imwrite(f"{video_dir}/{i:04d}.jpg", np.hstack([img_pred, img_gt]))
                cv2.imshow("pred-gt", np.hstack((img_pred, img_gt)))
                cv2.waitKey(20)
        
        square_error += (prediction - ground_truth).pow(2).sum()
        n += ground_truth.numel()
        pbar.update(1)
        pbar.set_postfix({"mse": (square_error / n).item()})

    pbar.close()
    
    return {"mse": square_error.cpu().numpy() / n}


class Evaluator:
    def __init__(self, model: ODPCModel, dc: DataConversion, env_configs: DictConfig, dataset_configs: DictConfig):
        self.model = model
        self.dc = dc
        self.env_configs = env_configs
        self.dataset_configs = dataset_configs
        self.device = next(model.parameters()).device
        
        # 初始化环境和agent
        self.envs = {}
        self.agents = {}
        
        if self.env_configs is not None:
            for env_name, env_config in self.env_configs.items():
                # 创建环境
                env_kwargs = OmegaConf.to_object(env_config.env.env_kwargs)
                other_kwargs = OmegaConf.to_object(env_config.env.other_kwargs)
                self.envs[env_name] = instantiate_from_config(
                    env_config.env,
                    env_kwargs=env_kwargs,
                    other_kwargs=other_kwargs,
                    wrappers=[FlattenRGBDObservationWrapper],
                )
               
                # 创建agent
                tmp_env = gym.make(env_config.env.env_id, **env_kwargs)
                origin_obs_space = tmp_env.observation_space
                tmp_env.close()
                self.agents[env_name] = instantiate_from_config(
                    env_config.agent,
                    model=self.model,
                    dc=self.dc,
                    origin_obs_space=origin_obs_space,
                    video_dir=env_config.agent.video_dir,
                ).eval().to(self.device)
        
        # 初始化数据集和对应的agent
        self.datasets = {}
        
        if self.dataset_configs is not None:
            for dataset_name, dataset_config in self.dataset_configs.items():
                # 创建数据集
                self.datasets[dataset_name] = instantiate_from_config(dataset_config)
                
             
    def evaluate(self, num_episodes: int = 100, batch_size: int = 256):
        """在环境和数据集上进行评估
        
        Args:
            num_episodes: 环境评估的episode数量
        """
        metrics = {}
        
        # 在环境上评估
        if self.env_configs is not None:
            for env_name, env_config in self.env_configs.items():
                env_metrics = evaluate_on_env(
                    num_episodes, self.agents[env_name], self.envs[env_name], 
                    self.device, env_config.env.sim_backend
                )
                metrics[f"env_{env_name}"] = env_metrics
        
        # 在数据集上评估
        if self.dataset_configs is not None:
            for dataset_name, dataset_config in self.dataset_configs.items():
                dataset_metrics = evaluate_on_dataset(
                    self.model, 
                    self.datasets[dataset_name], 
                    self.dc, 
                    batch_size, 
                    self.device,
                )
                metrics[f"dataset_{dataset_name}"] = dataset_metrics
        
        return metrics
        
    def close(self):
        """关闭所有环境"""
        for env in self.envs.values():
            env.close()
        
        