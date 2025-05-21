from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor, Pose
from mani_skill.envs.utils import randomization
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.agents.robots.panda import Panda
from mani_skill.agents.registration import register_agent


@register_agent()
class PandaForPegInsertion(Panda):
    uid = "panda_peg_insertion"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[0.954, 0, -0.3, 0]),
                width=256,
                height=256,
                fov=np.pi * 0.75,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ]


@register_env("PegInsertionSide-v2", max_episode_steps=100)
class PegInsertionSideV2Env(PegInsertionSideEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_peg_insertion", "xarm6_robotiq"]
    agent: Union[PandaForPegInsertion]
    _clearance = 0.005

    def __init__(
            self,
            *args,
            robot_uids="panda",
            camera_mode="fixed-2",  # fixed, random, move
            **kwargs,
    ):
        self.camera_mode = camera_mode
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        sapien.physx.set_default_material(static_friction=20.0, dynamic_friction=20.0, restitution=0.0)
        super()._load_scene(options)

    @property
    def _default_sensor_configs(self):
        if self.camera_mode == "fixed":
            pose = sapien_utils.look_at([0.4, -0.4, 0.4], [0.05, -0.1, 0.2])
        if self.camera_mode == "fixed-2":
            pose = sapien_utils.look_at([0.5, -0.5, 0.1], [0., 0., 0.1])
        elif self.camera_mode == "random":
            eye = np.random.uniform(low=[0.4, -0.4, 0.5], high=[0.7, -0.2, 0.8], size=(self.num_envs, 3))
            eye = torch.from_numpy(eye).float().to(device=self.device)

            target = np.random.uniform(low=[-0.1, 0, 0], high=[0.1, 0.3, 0.6], size=(self.num_envs, 3))
            target = torch.from_numpy(target).float().to(device=self.device)
            target[:, 2] *= eye[:, 2]

            pose = sapien_utils.look_at(eye, target, device=self.device)

        elif self.camera_mode == "move":
            raise NotImplementedError()

        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # initialize the box and peg
            xy = randomization.uniform(
                low=torch.tensor([-0.1, -0.3]), high=torch.tensor([0.1, 0]), size=(b, 2)
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.peg_half_sizes[env_idx, 2]
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 3, np.pi / 2 + np.pi / 3),
            )
            self.peg.set_pose(Pose.create_from_pq(pos, quat))

            xy = randomization.uniform(
                low=torch.tensor([-0.05, 0.2]),
                high=torch.tensor([0.05, 0.4]),
                size=(b, 2),
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.peg_half_sizes[env_idx, 0]
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
            )
            self.box.set_pose(Pose.create_from_pq(pos, quat))

            # Initialize the robot
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

    def _get_obs_extra(self, info: Dict):
        obs = super()._get_obs_extra(info)
        ros2opencv = Pose.create_from_pq(q=torch.tensor([0.5, 0.5, -0.5, 0.5]), device=self.peg.pose.device)
        cam0_world_pose = ros2opencv * self._sensors["base_camera"].camera.global_pose.inv()
        cam0_peg_pose = cam0_world_pose * self.peg.pose
        obs.update(
            base_pose=self.agent.robot.root.pose.raw_pose,
            cam0_peg_pose=cam0_peg_pose.raw_pose,
            cam0_world_pose=cam0_world_pose.raw_pose,
        )
        return obs
