import torch

from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_multiply, quaternion_apply,
)

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
