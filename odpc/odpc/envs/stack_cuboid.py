from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs import Actor


@register_env("StackCuboid-v1", max_episode_steps=50)
class StackCuboidEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to pick up a red cuboid and stack it on top of a green cuboid and let go of the cuboid without it falling

    **Randomizations:**
    - both cuboids have their z-axis rotation randomized
    - both cuboids have their xy positions on top of the table scene randomized. The positions are sampled such that the cuboids do not collide with each other
    - both cuboids have random heights between 4cm and 10cm

    **Success Conditions:**
    - the red cuboid is on top of the green cuboid (to within half of the cuboid size)
    - the red cuboid is static
    - the red cuboid is not being grasped by the robot (robot must let go of the cuboid)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, 
        *args, 
        robot_uids="panda", 
        robot_init_qpos_noise=0.02, 
        cuboid_half_size=0.02,
        reconfiguration_freq=None, 
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.cuboid_half_size = cuboid_half_size
        # 自动设置reconfiguration_freq，单环境时为1，多环境为0
        if reconfiguration_freq is None:
            if kwargs.get('num_envs', 1) == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(*args, robot_uids=robot_uids, reconfiguration_freq=reconfiguration_freq, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.4, 0.5, 0.5], [0.0, 0.0, 0.05])
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # 初始化时设置默认尺寸
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # 为每个环境随机生成长方体高度 (4cm-10cm)
        # 使用_batched_episode_rng为每个环境生成不同的随机高度
        heights_A = np.array([self._batched_episode_rng[i].uniform(0.02, 0.05) for i in range(self.num_envs)])  # 4cm-10cm
        heights_B = np.array([self._batched_episode_rng[i].uniform(0.02, 0.05) for i in range(self.num_envs)])  # 4cm-10cm
        heights_A = common.to_tensor(heights_A, device=self.device)
        heights_B = common.to_tensor(heights_B, device=self.device)
        
        # 保存每个环境的cuboid尺寸
        self.cuboidA_half_sizes = torch.stack([
            torch.ones(self.num_envs, device=self.device) * self.cuboid_half_size,  # 长4cm
            torch.ones(self.num_envs, device=self.device) * self.cuboid_half_size,  # 宽4cm
            heights_A  # 随机高度
        ], dim=1)
        
        self.cuboidB_half_sizes = torch.stack([
            torch.ones(self.num_envs, device=self.device) * self.cuboid_half_size,  # 长4cm
            torch.ones(self.num_envs, device=self.device) * self.cuboid_half_size,  # 宽4cm
            heights_B  # 随机高度
        ], dim=1)

        # 为每个并行环境创建不同尺寸的长方体
        cuboidsA = []
        cuboidsB = []

        for i in range(self.num_envs):
            scene_idxs = [i]
            
            # 创建长方体A
            builder = self.scene.create_actor_builder()
            half_sizes_A = [self.cuboid_half_size, self.cuboid_half_size, float(heights_A[i].item())]  # 长4cm，宽4cm，随机高度(float)
            builder.add_box_collision(half_size=half_sizes_A)
            builder.add_box_visual(
                half_size=half_sizes_A,
                material=sapien.render.RenderMaterial(
                    base_color=[1, 0, 0, 1],  # 红色
                    roughness=0.5,
                    specular=0.5,
                ),
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
            builder.set_scene_idxs(scene_idxs)
            cuboidA = builder.build(f"cuboidA_{i}")
            self.remove_from_state_dict_registry(cuboidA)
            
            # 创建长方体B
            builder = self.scene.create_actor_builder()
            half_sizes_B = [self.cuboid_half_size, self.cuboid_half_size, float(heights_B[i].item())]  # 长4cm，宽4cm，随机高度(float)
            builder.add_box_collision(half_size=half_sizes_B)
            builder.add_box_visual(
                half_size=half_sizes_B,
                material=sapien.render.RenderMaterial(
                    base_color=[0, 1, 0, 1],  # 绿色
                    roughness=0.5,
                    specular=0.5,
                ),
            )
            builder.initial_pose = sapien.Pose(p=[1, 0, 0.1])
            builder.set_scene_idxs(scene_idxs)
            cuboidB = builder.build(f"cuboidB_{i}")
            self.remove_from_state_dict_registry(cuboidB)
            
            cuboidsA.append(cuboidA)
            cuboidsB.append(cuboidB)
        
        # 合并所有环境的长方体
        self.cuboidA = Actor.merge(cuboidsA, "cuboidA")
        self.cuboidB = Actor.merge(cuboidsB, "cuboidB")
        
        # 注册到状态字典
        self.add_to_state_dict_registry(self.cuboidA)
        self.add_to_state_dict_registry(self.cuboidB)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # 设置位置
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = self.cuboidA_half_sizes[env_idx, 2]  # 使用实际高度作为z位置
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([self.cuboid_half_size, self.cuboid_half_size])) + 0.001
            cuboidA_xy = xy + sampler.sample(radius, 100)
            cuboidB_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cuboidA_xy
            xyz[:, 2] = self.cuboidA_half_sizes[env_idx, 2]
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cuboidA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cuboidB_xy
            xyz[:, 2] = self.cuboidB_half_sizes[env_idx, 2]
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cuboidB.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def evaluate(self):
        pos_A = self.cuboidA.pose.p
        pos_B = self.cuboidB.pose.p
        offset = pos_A - pos_B
        
        # 获取当前环境的长方体尺寸
        env_idx = torch.arange(len(pos_A), device=self.device)
        cuboidA_half_size = self.cuboidA_half_sizes[env_idx]
        cuboidB_half_size = self.cuboidB_half_sizes[env_idx]
        
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(cuboidA_half_size[:, :2], axis=1) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - (cuboidA_half_size[:, 2] + cuboidB_half_size[:, 2])) <= 0.005
        is_cuboidA_on_cuboidB = torch.logical_and(xy_flag, z_flag)
        
        is_cuboidA_static = self.cuboidA.is_static(lin_thresh=3e-2, ang_thresh=3e-1)
        is_cuboidA_grasped = self.agent.is_grasping(self.cuboidA)
        # success = is_cuboidA_on_cuboidB * is_cuboidA_static * (~is_cuboidA_grasped)
        success = is_cuboidA_on_cuboidB * (~is_cuboidA_grasped)
        return {
            "is_cuboidA_grasped": is_cuboidA_grasped,
            "is_cuboidA_on_cuboidB": is_cuboidA_on_cuboidB,
            "is_cuboidA_static": is_cuboidA_static,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cuboidA_pose=self.cuboidA.pose.raw_pose,
                cuboidB_pose=self.cuboidB.pose.raw_pose,
                tcp_to_cuboidA_pos=self.cuboidA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cuboidB_pos=self.cuboidB.pose.p - self.agent.tcp.pose.p,
                cuboidA_to_cuboidB_pos=self.cuboidB.pose.p - self.cuboidA.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        cuboidA_pos = self.cuboidA.pose.p
        cuboidA_to_tcp_dist = torch.linalg.norm(tcp_pose - cuboidA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cuboidA_to_tcp_dist))

        # grasp and place reward
        cuboidA_pos = self.cuboidA.pose.p
        cuboidB_pos = self.cuboidB.pose.p
        
        # 获取当前环境的长方体尺寸
        env_idx = torch.arange(len(cuboidA_pos), device=self.device)
        cuboidA_half_size = self.cuboidA_half_sizes[env_idx]
        cuboidB_half_size = self.cuboidB_half_sizes[env_idx]
        
        goal_xyz = torch.hstack(
            [cuboidB_pos[:, 0:2], (cuboidB_pos[:, 2] + cuboidA_half_size[:, 2] + cuboidB_half_size[:, 2])[:, None]]
        )
        cuboidA_to_goal_dist = torch.linalg.norm(goal_xyz - cuboidA_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cuboidA_to_goal_dist)

        reward[info["is_cuboidA_grasped"]] = (4 + place_reward)[info["is_cuboidA_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )  # NOTE: hard-coded with panda
        is_cuboidA_grasped = info["is_cuboidA_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_cuboidA_grasped] = 1.0
        v = torch.linalg.norm(self.cuboidA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cuboidA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info["is_cuboidA_on_cuboidB"]] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_cuboidA_on_cuboidB"]]

        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8 