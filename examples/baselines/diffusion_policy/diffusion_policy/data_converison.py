from typing import Literal, Optional, List

import numpy as np
import torch

from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_invert, quaternion_multiply, quaternion_apply,
    quaternion_to_matrix, matrix_to_quaternion,
    euler_angles_to_matrix, matrix_to_euler_angles,
    rotation_6d_to_matrix, matrix_to_rotation_6d,
    axis_angle_to_quaternion, quaternion_to_axis_angle,
)
from diffusion_policy.utils import visualize_pose

Frame = Literal["camera_first", "camera_cur", "world"]
RotationRepresentation = Literal["matrix_3x3", "rotation_6d", "quaternion", "euler", "axis_angle", "se3"]
DeltaPrediction = Literal["abs", "rel_to_first", "rel_to_prev"]


def pose_inv(pose: torch.Tensor) -> torch.Tensor:
    new_pose = pose.clone()
    new_pose[..., 4:] = -pose[..., 4:]
    new_pose[..., :3] = quaternion_apply(new_pose[..., 3:], -pose[..., :3])
    return new_pose


def _pose_multiply(pose1: torch.Tensor, pose2: torch.Tensor) -> torch.Tensor:
    p1, q1 = pose1[..., :3], pose1[..., 3:]
    p2, q2 = pose2[..., :3], pose2[..., 3:]
    new_q = quaternion_multiply(q1, q2)
    new_p = p1 + quaternion_apply(q1, p2)
    return torch.cat([new_p, new_q], dim=-1)


def pose_multiply(*args: torch.Tensor) -> torch.Tensor:
    if len(args) == 2:
        return _pose_multiply(args[0], args[1])
    return _pose_multiply(args[0], pose_multiply(*args[1:]))


class DataConversion:
    def __init__(
            self,
            frame: Frame = "camera_first",
            delta_pred: DeltaPrediction = "rel_to_prev",
            rot_rep: RotationRepresentation = "axis_angle",
            control_mode: str = "pd_ee_delta_pose",
            dt=0.05,
    ):
        self.frame = frame
        self.delta_pred = delta_pred
        self.rot_rep = rot_rep
        self.dt = dt
        self.control_mode = control_mode

    @property
    def pred_dim(self) -> int:
        if self.delta_pred == "abs":
            return 7
        else:
            return 6

    @property
    def control_dim(self) -> int:
        return 6

    def raw_to_pred(
            self,
            poses_obj: torch.Tensor,
            poses_camera_world: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            poses_obj: object pose in world, tensor of shape (..., T+1, 7)
            poses_camera_world: world in camera, tensor of shape (..., T+1, 7)

        Returns:
            pred: nn prediction, tensor of shape (..., T, D)
        """
        if self.frame == "camera_first":
            # T_cam,0_obj,t = T_cam,0_world * T_world_obj,t
            poses_obj = pose_multiply(poses_camera_world[..., :1, :], poses_obj)
        elif self.frame == "camera_cur":
            # T_cam,t_obj,t = T_cam,t_world * T_world_obj,t
            poses_obj = pose_multiply(poses_camera_world, poses_obj)
        else:
            raise NotImplementedError

        if self.delta_pred == "abs":
            poses_obj = poses_obj[..., 1:, :]
        elif self.delta_pred == "rel_to_first":
            # Delta_1(t) = T_cam_obj,t * inv( T_cam_obj,0 )
            poses_obj = pose_multiply(poses_obj[..., 1:, :], pose_inv(poses_obj[..., :1, :]))
        elif self.delta_pred == "rel_to_prev":
            # Delta_2(t) = T_cam_obj,t * inv( T_cam_obj,t-1 )
            poses_obj = pose_multiply(poses_obj[..., 1:, :], pose_inv(poses_obj[..., :-1, :]))

        pos = poses_obj[..., :3]
        rot = poses_obj[..., 3:]
        if self.rot_rep == "matrix_3x3":
            rot = quaternion_to_matrix(rot)
            rot = rot.flatten(start_dim=-2)
        elif self.rot_rep == "rotation_6d":
            rot = matrix_to_rotation_6d(quaternion_to_matrix(rot))
        elif self.rot_rep == "quaternion":
            pass
        elif self.rot_rep == "euler":
            rot = matrix_to_euler_angles(quaternion_to_matrix(rot), "XYZ")
            rot = rot / self.dt
        elif self.rot_rep == "axis_angle":
            rot = quaternion_to_axis_angle(rot)
            rot = rot / self.dt
        else:
            raise NotImplementedError

        return torch.cat((pos, rot), dim=-1)

    @staticmethod
    def expend_dim_to(a, dim, length):
        if a.shape[dim] < length:
            expand = torch.repeat_interleave(torch.narrow(a, dim, -1, 1), length - a.shape[dim], dim)
            a = torch.cat((a, expand), dim=dim)
        return a

    @torch.no_grad()
    def pred_to_control(
            self,
            pred: torch.Tensor,
            poses_ee_cur: torch.Tensor,
            poses_base: torch.Tensor,
            poses_camera: Optional[torch.Tensor] = None,
            poses_camera_world: Optional[torch.Tensor] = None,
            poses_frame_obj_cur: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: nn prediction, tensor of shape (..., T, D)
            poses_ee_cur: ee in world, tensor of shape (..., 1, 7)
            poses_base: robot base in world, tensor of shape (..., T+1, 7)
            poses_camera: camera in world, tensor of shape (..., T+1, 7)
            poses_camera_world: world in camera, tensor of shape (..., T+1, 7)
            poses_frame_obj_cur: current object poses, assume in the self.frame, tensor of shape (..., 1, 7)

        Returns:
            tensor of shape (..., T, 6) for "pd_ee_delta_pose", (..., T, 7) for "pd_ee_pose"
        """
        t = pred.shape[-2]
        poses_base = self.expend_dim_to(poses_base, -2, t + 1)

        assert poses_camera is not None or poses_camera_world is not None
        if poses_camera is None:
            poses_camera_world = self.expend_dim_to(poses_camera_world, -2, t + 1)
            poses_camera = pose_inv(poses_camera_world)
        if poses_camera_world is None:
            poses_camera = self.expend_dim_to(poses_camera, -2, t + 1)
            poses_camera_world = pose_inv(poses_camera)

        pred = self.pred_to_quaternion(pred)

        if self.delta_pred == "abs":
            pred = pose_multiply(pred, pose_inv(poses_frame_obj_cur))
        elif self.delta_pred == "rel_to_first":
            pass
        elif self.delta_pred == "rel_to_prev":
            # Delta_1(t) = Delta_2(t) * Delta_2(t-1) * ... * Delta_2(1)
            t = pred.shape[-2]
            pred = pred.clone()
            for i in range(1, t):
                pred[..., i, :] = pose_multiply(pred[..., i, :], pred[..., i - 1, :])
        else:
            raise NotImplementedError

        if self.frame == "camera_first":
            # T_base,t_cam,0 = inv( T_world_base,t ) * T_world_cam,0
            poses_base_cam = pose_multiply(pose_inv(poses_base[..., 1:, :]), poses_camera[..., :1, :])
            # T_cam,0_base,0 = inv( T_world_cam,0 ) * T_world_base,0
            poses_cam_base = pose_multiply(poses_camera_world[..., :1, :], poses_base[..., :1, :])
        elif self.frame == "camera_cur":
            # T_base,t_cam,t = inv( T_world_base,t ) * T_world_cam,0
            poses_base_cam = pose_multiply(pose_inv(poses_base[..., 1:, :]), poses_camera[..., 1:, :])
            # T_cam,0_base,0 = inv( T_world_cam,0 ) * T_world_base,0
            poses_cam_base = pose_multiply(poses_camera_world[..., :1, :], poses_base[..., :1, :])
        else:
            raise NotImplementedError

        poses_base_ee_cur = pose_multiply(pose_inv(poses_base[..., :1, :]), poses_ee_cur)
        poses_base_ee = pose_multiply(poses_base_cam, pred, poses_cam_base, poses_base_ee_cur)

        if self.control_mode == "pd_ee_pose":
            p = poses_base_ee[..., :3]
            q = poses_base_ee[..., 3:]
            euler = matrix_to_euler_angles(quaternion_to_matrix(q), "XYZ")
            return torch.cat((p, euler), dim=-1)
        elif self.control_mode == "pd_ee_delta_pose":
            poses_base_eff = torch.cat((poses_base_ee_cur, poses_base_ee), dim=-2)
            p = poses_base_eff[..., :3]
            q = poses_base_eff[..., 3:]
            delta_p = (p[..., 1:, :] - p[..., :-1, :]) / self.dt
            delta_q = quaternion_multiply(q[..., 1:, :], quaternion_invert(q[..., :-1, :]))
            euler = matrix_to_euler_angles(quaternion_to_matrix(delta_q), "XYZ") / self.dt
            return torch.cat((delta_p, -euler), dim=-1)
        else:
            raise NotImplementedError


    @torch.no_grad()
    def pred_to_visualize(
            self,
            rgb: torch.Tensor,
            pred: torch.Tensor,
            intrinsic: Optional[torch.Tensor] = None,
            poses_cam_obj_cur: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Args:
            rgb: RGB image, tensor of shape (..., 1, C, H, W)
            pred: nn prediction, tensor of shape (..., T, D)
            intrinsic: intrinsic matrix, tensor of shape (3, 3)
            poses_cam_obj_cur: current object poses in camera, tensor of shape (..., 1, 7)

        Returns:
            tensor of shape (..., T, 6) for "pd_ee_delta_pose", (..., T, 7) for "pd_ee_pose"
        """
        assert self.frame in ["camera_first", "camera_cur"]
        assert pred.ndim == 3

        rgb = rgb.squeeze(dim=-4)
        dims = list(range(rgb.ndim))
        rgb = rgb.permute(*dims[:-3], -2, -1, -3)
        rgb = rgb.cpu().detach().numpy()

        if intrinsic is None:
            h, w = rgb.shape[-3:-1]
            h, w = h / 2, w / 2
            intrinsic = torch.tensor([
                [w, 0, w],
                [0, h, h],
                [0, 0, 1],
            ])

        pred = self.pred_to_quaternion(pred)

        if self.delta_pred == "abs":
            poses_cam_obj = pred
        elif self.delta_pred == "rel_to_first":
            poses_cam_obj = pose_multiply(pred, poses_cam_obj_cur)
        elif self.delta_pred == "rel_to_prev":
            # Delta_1(t) = Delta_2(t) * Delta_2(t-1) * ... * Delta_2(1)
            t = pred.shape[-2]
            pred = pred.clone()
            for i in range(1, t):
                pred[..., i, :] = pose_multiply(pred[..., i, :], pred[..., i - 1, :])
            poses_cam_obj = pose_multiply(pred, poses_cam_obj_cur)
        else:
            raise NotImplementedError

        poses_cam_obj = torch.cat((poses_cam_obj_cur, poses_cam_obj), dim=-2)

        n, t = poses_cam_obj.shape[:2]
        images = []
        for i in range(n):
            image = rgb[i]
            for j in range(t):
                pose = poses_cam_obj[i, j]
                alpha = (j + 5) / (t + 4)
                image = image * (1 - alpha) + visualize_pose(image, pose, intrinsic) * alpha
            images.append(image.astype(np.uint8))

        return np.stack(images, 0)

    def pred_to_quaternion(self, pred):
        pos = pred[..., :3]
        rot = pred[..., 3:]

        if self.rot_rep == "matrix_3x3":
            # TODO: 如果不满足 RR^T = I,
            rot = rot.reshape(rot.shape[:-1] + (3, 3))
            rot = matrix_to_quaternion(rot)
        elif self.rot_rep == "rotation_6d":
            rot = matrix_to_quaternion(rotation_6d_to_matrix(rot))
        elif self.rot_rep == "quaternion":
            pass
        elif self.rot_rep == "euler":
            rot = rot * self.dt
            rot = matrix_to_quaternion(euler_angles_to_matrix(rot, "XYZ"))
        elif self.rot_rep == "axis_angle":
            rot = rot * self.dt
            rot = axis_angle_to_quaternion(rot)
        else:
            raise NotImplementedError

        return torch.cat((pos, rot), dim=-1)
