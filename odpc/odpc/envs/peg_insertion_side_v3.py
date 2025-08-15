from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.utils import randomization
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose

from .peg_insertion_side_v2 import PegInsertionSideV2Env


# 从 mani_skill.envs.tasks.tabletop.peg_insertion_side 复制而来
def _build_box_with_hole(
    scene: ManiSkillScene, inner_radius, outer_radius, depth, center=(0, 0)
):
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    # x-axis is hole direction
    half_center = [x * 0.5 for x in center]
    half_sizes = [
        [depth, thickness - half_center[0], outer_radius],
        [depth, thickness + half_center[0], outer_radius],
        [depth, outer_radius, thickness - half_center[1]],
        [depth, outer_radius, thickness + half_center[1]],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([0, offset + half_center[0], 0]),
        sapien.Pose([0, -offset + half_center[0], 0]),
        sapien.Pose([0, 0, offset + half_center[1]]),
        sapien.Pose([0, 0, -offset + half_center[1]]),
    ]

    mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
    )

    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)
    return builder


@register_env("PegInsertionSide-v3", max_episode_steps=100)
class PegInsertionSideV3Env(PegInsertionSideV2Env):
    def __init__(self, *args, robot_uids="panda", **kwargs):
        # 根据 v3 的规范，间隙是固定的。
        # Peg（插头）的横截面为 0.03x0.03 -> 半径 0.015
        # Hole（孔）的尺寸为 0.06x0.06 -> 半径 0.03
        # 间隙 = 孔半径 - 插头半径 = 0.015
        super().__init__(*args, robot_uids=robot_uids, clearance=0.015, **kwargs)

    def _load_scene(self, options: dict):
        # 重写方法，以使用 v3 规范中的固定尺寸。
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            # 来自 v3 规范的固定尺寸 (移植自 metaworld)
            peg_half_length = 0.12
            peg_half_width = 0.015
            box_half_depth = 0.1
            box_half_width = 0.1

            lengths = torch.full(
                (self.num_envs,), peg_half_length, device=self.device
            )
            radii = torch.full((self.num_envs,), peg_half_width, device=self.device)

            # 孔位于盒子表面的中心
            centers = torch.zeros((self.num_envs, 2), device=self.device)

            self.peg_half_sizes = torch.vstack([lengths, radii, radii]).T
            peg_head_offsets = torch.zeros((self.num_envs, 3), device=self.device)
            peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
            self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

            box_hole_offsets = torch.zeros((self.num_envs, 3), device=self.device)
            box_hole_offsets[:, 1:] = centers
            self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)

            self.box_hole_radii = radii + self._clearance

            pegs = []
            boxes = []

            for i in range(self.num_envs):
                scene_idxs = [i]
                length = self.peg_half_sizes[i, 0]
                radius = self.peg_half_sizes[i, 1]

                length_val = length.item()
                radius_val = radius.item()

                # 构建 peg (插头)
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[length_val, radius_val, radius_val])
                mat_head = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EC7357"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([length_val / 2, 0, 0]),
                    half_size=[length_val / 2, radius_val, radius_val],
                    material=mat_head,
                )
                mat_tail = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EDF6F9"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([-length_val / 2, 0, 0]),
                    half_size=[length_val / 2, radius_val, radius_val],
                    material=mat_tail,
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, 1.0])
                builder.set_scene_idxs(scene_idxs)
                peg = builder.build(f"peg_{i}")
                self.remove_from_state_dict_registry(peg)

                # 构建带孔的盒子
                inner_radius, outer_radius, depth = (
                    self.box_hole_radii[i],
                    box_half_width,
                    box_half_depth,
                )
                builder = _build_box_with_hole(
                    self.scene,
                    inner_radius.item(),
                    outer_radius,
                    depth,
                    center=centers[i].cpu().numpy(),
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, 1.0])
                builder.set_scene_idxs(scene_idxs)
                box = builder.build_kinematic(f"box_with_hole_{i}")
                self.remove_from_state_dict_registry(box)

                pegs.append(peg)
                boxes.append(box)

            self.peg = Actor.merge(pegs, "peg")
            self.box = Actor.merge(boxes, "box_with_hole")

            self.add_to_state_dict_registry(self.peg)
            self.add_to_state_dict_registry(self.box)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # 随机区域的尺寸参考自 metaworld，但区域中心沿用 V2 的设定以确保不出界

            # Peg 在以 (0, -0.15) 为中心的 0.2x0.2 区域内随机生成
            peg_xy = randomization.uniform(
                low=torch.tensor([-0.1, -0.25], device=self.device),
                high=torch.tensor([0.1, -0.05], device=self.device),
                size=(b, 2),
            )
            peg_pos = torch.zeros((b, 3), device=self.device)
            peg_pos[:, :2] = peg_xy
            peg_pos[:, 2] = 0.015 # Peg 半径，确保其在桌面上

            # 绕 Z 轴逆时针旋转90度(+pi/2)，使其朝向+Y方向
            rot_z_plus_90 = torch.tensor([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], dtype=torch.float32, device=self.device)
            peg_quat = rot_z_plus_90.repeat(b, 1)
            self.peg.set_pose(Pose.create_from_pq(p=peg_pos, q=peg_quat))

            # Box 在以 (0, 0.3) 为中心的 0.3x0.1 区域内随机生成
            box_xy = randomization.uniform(
                low=torch.tensor([-0.15, 0.25], device=self.device),
                high=torch.tensor([0.15, 0.35], device=self.device),
                size=(b, 2),
            )
            box_pos = torch.zeros((b, 3), device=self.device)
            box_pos[:, :2] = box_xy
            box_pos[:, 2] = 0.1 # Box 半高，确保其在桌面上

            # 同样绕 Z 轴逆时针旋转90度(+pi/2)，使其与 peg 朝向相同
            box_quat = rot_z_plus_90.repeat(b, 1)
            self.box.set_pose(Pose.create_from_pq(p=box_pos, q=box_quat))

            # 初始化机器人 - 与 v2 相同
            if "panda" in self.robot_uids:
                qpos = np.array(
                    [
                        0.0,
                        np.pi / 8,
                        0,
                        -np.pi * 5 / 8,
                        0,
                        np.pi * 3 / 4,
                        -np.pi / 4,
                        0.04,
                        0.04,
                    ]
                )
                qpos = self._episode_rng.normal(0, 0.02, (b, len(qpos))) + qpos
                qpos[:, -2:] = 0.04
                self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
