import os
import cv2
import copy
import trimesh.primitives
from gymnasium.spaces import Dict, Box
from omegaconf import DictConfig, ListConfig
from typing import List

import torch
import torch.nn as nn
import numpy as np

import sapien.core as sapien

from mani_skill.utils.structs import Pose
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_multiply, quaternion_invert, matrix_to_euler_angles, quaternion_to_matrix,
    matrix_to_quaternion, euler_angles_to_matrix
)

from odpc.data.data_conversion import DataConversion
from odpc.models.policy.base_policy import BasePolicy
from odpc.utils.math import pose_multiply, pose_inv
from odpc.utils.visualize import visualize_pose
from odpc.models.agent.base_agent import BaseAgent
from odpc.utils.utils import instantiate_from_config

DT = 0.05
DP_MAX = DT * 0.5
DQ_MAX = DT * 1.0


class ODPCAgent(BaseAgent):
    def __init__(
            self,
            model: BasePolicy,
            num_envs: int,
            act_horizon: int,
            pred_horizon: int,
            control_mode: str = 'pd_ee_pose',
            robot_uid: str = 'panda',
            video_dir: str = None,
            dc_configs: DictConfig = {},
            obs_processor_configs: ListConfig = [],
    ):
        super().__init__(
            model=model,
            num_envs=num_envs,
            act_horizon=act_horizon,
            pred_horizon=pred_horizon,
            video_dir=video_dir,
            obs_processor_configs=obs_processor_configs,
        )
        self.dc: DataConversion = instantiate_from_config(dc_configs)
        self.control_mode = control_mode
        self.video_dir = video_dir

        self._video_writer = None
        if video_dir is not None:
            os.makedirs(video_dir, exist_ok=True)
            
        self.stages = torch.zeros(num_envs)
        self.model_action = torch.zeros((num_envs, pred_horizon, self.dc.control_dim))

        self.base_grasp_pose = torch.zeros((num_envs, 7))
        self.base_reach_pose = torch.zeros((num_envs, 7))

        if "panda" in robot_uid:
            self.gripper_state = torch.tensor([1., -1.])  # open, close
        elif "robotiq" in robot_uid:
            self.gripper_state = torch.tensor([0., 0.81])  # open, close
        else:
            raise NotImplementedError

    
    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose ()."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    def get_grasp_pose(
            self,
            peg_half_size: torch.Tensor,
            peg_pose: torch.Tensor,
            ee_pose: torch.Tensor,
            base_pose: torch.Tensor,
    ):
        for i in range(self.num_envs):
            extents = peg_half_size[i].cpu().numpy() * 2
            transform = (
                    Pose.create(peg_pose[i]) * sapien.Pose([-0.07, 0, 0])
            ).to_transformation_matrix()[0].cpu().numpy()
            obb = trimesh.primitives.Box(extents=extents, transform=transform)
            approaching = np.array([0, 0, -1])
            target_closing = Pose.create(ee_pose[i]).to_transformation_matrix()[0, :3, 1].cpu().numpy()
            grasp_info = compute_grasp_info_by_obb(
                obb, approaching=approaching, target_closing=target_closing, depth=0.025)
            closing, center = grasp_info["closing"], grasp_info["center"]
            grasp_pose = self.build_grasp_pose(approaching, closing, center)
            self.base_grasp_pose[i, :3] = torch.from_numpy(grasp_pose.p)
            self.base_grasp_pose[i, 3:] = torch.from_numpy(grasp_pose.q)
            reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
            self.base_reach_pose[i, :3] = torch.from_numpy(reach_pose.p)
            self.base_reach_pose[i, 3:] = torch.from_numpy(reach_pose.q)
        self.base_grasp_pose = pose_multiply(pose_inv(base_pose), self.base_grasp_pose.to(base_pose))
        self.base_reach_pose = pose_multiply(pose_inv(base_pose), self.base_reach_pose.to(base_pose))


    def target_pose_to_action(
            self, 
            target_pose: torch.Tensor, 
            base_pose: torch.Tensor, 
            ee_pose: torch.Tensor, 
            target_in_base: bool = False,
    ):
        if target_in_base:
            base_target = target_pose
        else:
            base_target = pose_multiply(pose_inv(base_pose), target_pose)
        base_ee = pose_multiply(pose_inv(base_pose), ee_pose)

        dp = base_target[..., :3] - base_ee[..., :3]
        dq = quaternion_multiply(base_target[..., 3:], quaternion_invert(base_ee[..., 3:]))
        de = matrix_to_euler_angles(quaternion_to_matrix(dq), "XYZ")

        dp = dp * torch.clamp(DP_MAX / (torch.norm(dp, dim=-1, keepdim=True) + 1e-6), max=1.0)
        de = de * torch.clamp(DQ_MAX / (torch.norm(de, dim=-1, keepdim=True) + 1e-6), max=1.0)

        if self.control_mode == "pd_ee_delta_pose":
            action = torch.cat([dp, -de], dim=-1)
        elif self.control_mode == "pd_ee_pose":
            p = base_ee[..., :3] + dp
            dq = matrix_to_quaternion(euler_angles_to_matrix(de, "XYZ"))
            q = quaternion_multiply(dq, base_ee[..., 3:])
            euler = matrix_to_euler_angles(quaternion_to_matrix(q), "XYZ")
            action = torch.cat([p, euler], dim=-1)
        else:
            raise NotImplementedError

        return action


    def grasp(self, obs):
        ee_pose = obs["extra"]["tcp_pose"][:, 0, :]
        base_pose = obs["extra"]["base_pose"][:, 0, :]
        base_ee_pose = pose_multiply(pose_inv(base_pose), ee_pose)

        if self.stages[0] == 0:
            self.get_grasp_pose(
                peg_half_size=obs['extra']['peg_half_size'][:, -1, :],
                peg_pose=obs['extra']['peg_pose'][:, -1, :],
                ee_pose=ee_pose,
                base_pose=base_pose,
            )
            self.stages[:] = 1
        
        # get mp_action (motion-planning action)
        def reach_target(threshold):
            dp = torch.sum(torch.abs(base_ee_pose[i, :3] - mp_target_pose[i, :3])).item()
            dq1 = torch.sum(torch.abs(base_ee_pose[i, 3:] - mp_target_pose[i, 3:])).item()
            dq2 = torch.sum(torch.abs(base_ee_pose[i, 3:] + mp_target_pose[i, 3:])).item()
            return dp + min(dq1, dq2) < threshold

        mp_target_pose = torch.zeros_like(ee_pose)

        for i in range(self.num_envs):
            if self.stages[i] == 1:
                mp_target_pose[i] = self.base_reach_pose[i]
                if reach_target(0.05):
                    self.stages[i] = 2
            if self.stages[i] == 2:
                mp_target_pose[i] = self.base_grasp_pose[i]
                if reach_target(0.02):
                    self.stages[i] = 3
            if 3 <= self.stages[i] < 4:
                mp_target_pose[i] = self.base_grasp_pose[i]
                self.stages[i] += 1 / 20
            if 4 <= self.stages[i] < 5:
                mp_target_pose[i] = self.base_reach_pose[i]
                if reach_target(0.05):
                    self.stages[i] = 5
        
        return mp_target_pose

    @staticmethod
    def make_grid(images: np.ndarray) -> np.ndarray:
        n = len(images)
        nh = int(np.sqrt(n))
        nw = int(np.ceil(n / nh))
        h, w = images.shape[1:3]
        grid = np.zeros((nh * h, nw * w, 3), np.uint8)
        for i, rgb in enumerate(images):
            x, y = (i % nh) * h, (i // nh) * w
            grid[x: x + h, y: y + w] = rgb
        return grid

    @torch.no_grad()
    def odpc_exec(self, obs):
        ee_pose = obs["extra"]["tcp_pose"][:, 0, :]
        base_pose = obs["extra"]["base_pose"][:, 0, :]
        cam_obj_pose_cur = obs["extra"]["cam0_peg_pose"][..., :1, :]

        # Receding Horizon Control
        action_step = self._action_step % self.act_horizon

        if action_step == 0:
            obs_processed = copy.deepcopy(obs)
            obs_processed = self.permute_obs(obs_processed, [0, 1, 3, 4, 2])
            for processor in self.obs_processors:
                obs_processed = processor.process(obs_processed)
            obs_processed = self.permute_obs(obs_processed, [0, 1, 4, 2, 3])

            pred = self.model.get_action(obs_processed)

            self.odpc_target_pose = self.dc.pred_to_target_pose(
                pred=pred.clone().detach(),
                poses_ee_cur=obs["extra"]["tcp_pose"][..., :1, :],
                poses_base=obs["extra"]["base_pose"][..., :1, :],
                poses_camera_world=obs["extra"]["cam0_world_pose"][..., :1, :],
            )
            if self.video_dir is not None:
                self.cam_obj_pose = self.dc.pred_to_cam_obj_pose(
                    pred=pred.clone().detach(),
                    poses_cam_obj_cur=cam_obj_pose_cur,
                )

        if self.video_dir is not None:
            cam_obj_pose = self.cam_obj_pose[..., action_step:, :]
            
            rgb = obs["sensor_data"]["base_camera"]["rgb"][:, 0, :3, :, :].cpu().numpy()
            rgb = rgb.transpose(0, 2, 3, 1)
            
            h, w = rgb.shape[-3:-1]
            h, w = h / 2, w / 2
            intrinsic = np.array([
                [w, 0, w],
                [0, h, h],
                [0, 0, 1],
            ])
            
            n, t = cam_obj_pose.shape[:2]
            images = []
            for i in range(n):
                image = rgb[i]
                for j in range(t):
                    image = visualize_pose(image, cam_obj_pose[i, j].cpu().numpy(), intrinsic)
                image = visualize_pose(image, cam_obj_pose_cur[i, 0].cpu().numpy(), intrinsic)
                images.append(image.astype(np.uint8))
            images = np.stack(images, axis=0)
            
            grid = self.make_grid(images)
            self._video_writer.write(grid[:, :, ::-1])

        return self.odpc_target_pose[..., action_step, :]

    @torch.no_grad()
    def get_action(self, obs, channel_last=False):
        if channel_last:
            obs = self.permute_obs(obs, [0, 1, 4, 2, 3])
        
        # ee 在 base 坐标系下的 target 坐标
        mp_target_pose = self.grasp(obs)
        odpc_target_pose = self.odpc_exec(obs)
        cond = self.stages.to(odpc_target_pose.device) == 5
        target_pose = torch.where(cond[:, None], odpc_target_pose, mp_target_pose)

        action = self.target_pose_to_action(
            target_pose=target_pose,
            base_pose=obs["extra"]["base_pose"][..., 0, :],
            ee_pose=obs["extra"]["tcp_pose"][..., 0, :],
            target_in_base=True,
        )
        
        # add gripper action
        gripper = self.gripper_state[(self.stages >= 3).int()].to(action)
        action = torch.cat([action, gripper[:, None]], dim=-1)

        self._action_step += 1

        return action[:, None, :]

    def reset(self, obs=None, channel_last=False):
        self.stages[:] = 0
        self._action_step = 0

        if self._video_writer is not None:
            self._video_writer.release()

        if self.video_dir is not None:
            # 搜索视频目录下所有视频文件，找到ID最大的那个，+1后设置为当前视频文件的ID
            video_files = os.listdir(self.video_dir)
            video_files = [int(file.split('.')[0]) for file in video_files]
            video_id = max(video_files) + 1 if len(video_files) > 0 else 0
            rgb = obs["sensor_data"]["base_camera"]["rgb"]
            h, w = rgb.shape[-3:-1] if channel_last else rgb.shape[-2:]
            nh = int(np.sqrt(self.num_envs))
            nw = int(np.ceil(self.num_envs / nh))
            h = h * nh
            w = w * nw
            self._video_writer = cv2.VideoWriter(
                f"{self.video_dir}/{video_id:04d}.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                20,
                (w, h),
            )
    
    def close(self):
        if self._video_writer is not None:
            self._video_writer.release()
