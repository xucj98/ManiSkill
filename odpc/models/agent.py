import os
import cv2
import trimesh.primitives
from gymnasium.spaces import Dict, Box

import torch
import torch.nn as nn
import numpy as np

import sapien.core as sapien

from mani_skill import Pose
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_multiply, quaternion_invert, matrix_to_euler_angles, quaternion_to_matrix
)

from odpc.data.data_conversion import DataConversion
from odpc.models.odpc import ODPCModel
from odpc.utils.math import pose_multiply, pose_inv



class ODPCAgent(nn.Module):
    def __init__(
            self,
            model: ODPCModel,
            dc: DataConversion,
            origin_obs_space: Dict,
            num_envs: int,
            act_horizon: int,
            pred_horizon: int,
            control_mode: str = 'pd_ee_pose',
            robot_uid: str = 'panda',
            video_dir: str = None,
    ):
        super(ODPCAgent, self).__init__()
        self.model = model
        self.dc = dc
        self.num_envs = num_envs
        self.origin_obs_space = origin_obs_space
        self.act_horizon = act_horizon
        self.control_mode = control_mode
        self.video_dir = video_dir

        if video_dir is not None:
            os.makedirs(video_dir, exist_ok=True)

        self.stages = torch.zeros(num_envs)
        self.action_step = 0
        self.model_action = torch.zeros((num_envs, pred_horizon, dc.control_dim))

        self.grasp_pose = torch.zeros((num_envs, 7))
        self.reach_pose = torch.zeros((num_envs, 7))

        if "panda" in robot_uid:
            self.gripper_state = torch.tensor([1., -1.])  # open, close
        elif "robotiq" in robot_uid:
            self.gripper_state = torch.tensor([0., 0.81])  # open, close
        else:
            raise NotImplementedError

    def state_to_dict(self, state, ref_dict):
        state_dict = {}
        for k, v in ref_dict.items():
            if k in ['sensor_data', 'sensor_param']:
                continue
            if isinstance(v, Dict):
                state, state_dict[k] = self.state_to_dict(state, v)
            elif isinstance(v, Box):
                state_dict[k] = state[..., :v.shape[-1]]
                state = state[..., v.shape[-1]:]
        return state, state_dict

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
            peg_half_size,
            peg_pose,
            ee_pose
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
            self.grasp_pose[i, :3] = torch.from_numpy(grasp_pose.p)
            self.grasp_pose[i, 3:] = torch.from_numpy(grasp_pose.q)
            reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
            self.reach_pose[i, :3] = torch.from_numpy(reach_pose.p)
            self.reach_pose[i, 3:] = torch.from_numpy(reach_pose.q)

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

    def get_action(self, obs_seq, channel_last=False):
        _, state_dict = self.state_to_dict(obs_seq["state"], self.origin_obs_space)
        if channel_last:
            obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)       
        
        ee_pose = state_dict["extra"]["tcp_pose"][:, 0, :]
        base_pose = state_dict["extra"]["base_pose"][:, 0, :]

        if self.action_step % self.act_horizon == 0:
            pred = self.model.get_action(obs_seq)
            self.model_action = self.dc.pred_to_control(
                pred=pred.clone().detach(),
                poses_ee_cur=state_dict["extra"]["tcp_pose"][..., :1, :],
                poses_base=state_dict["extra"]["base_pose"][..., :1, :],
                poses_camera_world=state_dict["extra"]["cam0_world_pose"][..., :1, :],
            )

            if self.video_dir is not None:
                images = self.dc.pred_to_visualize(
                    rgb=obs_seq["rgb"][..., :3, :, :],
                    pred=pred.clone().detach(),
                    poses_cam_obj_cur=state_dict["extra"]["cam0_peg_pose"][..., :1, :],
                )
                grid = self.make_grid(images)
                cv2.imwrite(f"{self.video_dir}/{self.action_step:04d}.jpg", grid[:, :, ::-1])

        if self.stages[0] == 0:
            self.get_grasp_pose(
                peg_half_size=state_dict['extra']['peg_half_size'][:, -1, :],
                peg_pose=state_dict['extra']['peg_pose'][:, -1, :],
                ee_pose=ee_pose,
            )
            self.stages[:] = 1

        # get mp_action (motion-planning action)
        def reach_target(threshold):
            dp = torch.sum(torch.abs(ee_pose[i, :3] - mp_target_pose[i, :3])).item()
            dq1 = torch.sum(torch.abs(ee_pose[i, 3:] - mp_target_pose[i, 3:])).item()
            dq2 = torch.sum(torch.abs(ee_pose[i, 3:] + mp_target_pose[i, 3:])).item()
            return dp + min(dq1, dq2) < threshold

        mp_target_pose = torch.zeros_like(ee_pose)

        for i in range(self.num_envs):
            if self.stages[i] == 1:
                mp_target_pose[i] = self.reach_pose[i]
                if reach_target(0.05):
                    self.stages[i] = 2
            if self.stages[i] == 2:
                mp_target_pose[i] = self.grasp_pose[i]
                if reach_target(0.02):
                    self.stages[i] = 3
            if 3 <= self.stages[i] < 4:
                mp_target_pose[i] = self.grasp_pose[i]
                self.stages[i] += 1 / 20
            if 4 <= self.stages[i] < 5:
                mp_target_pose[i] = self.reach_pose[i]
                if reach_target(0.05):
                    self.stages[i] = 5
        base_target = pose_multiply(pose_inv(base_pose), mp_target_pose)
        if self.control_mode == "pd_ee_delta_pose":
            base_ee = pose_multiply(pose_inv(base_pose), ee_pose)
            p = base_target[..., :3] - base_ee[..., :3]
            q = quaternion_multiply(base_target[..., 3:], quaternion_invert(base_ee[..., 3:]))
            euler = matrix_to_euler_angles(quaternion_to_matrix(q), "XYZ")
            mp_action = torch.cat([p, -euler], dim=-1)
        elif self.control_mode == "pd_ee_pose":
            p = base_target[..., :3]
            q = base_target[..., 3:]
            euler = matrix_to_euler_angles(quaternion_to_matrix(q), "XYZ")
            mp_action = torch.cat([p, euler], dim=-1)
        else:
            raise NotImplementedError

        # use agent action
        for i in range(self.num_envs):
            if self.stages[i] == 5:
                mp_action[i] = self.model_action[i][self.action_step % self.act_horizon]

        gripper = self.gripper_state[(self.stages >= 3).int()].to(mp_action)
        action = torch.cat([mp_action, gripper[:, None]], dim=-1)

        self.action_step += 1

        return action[:, None, :]

    def reset(self, obs=None):
        self.stages[:] = 0
        self.action_step = 0
