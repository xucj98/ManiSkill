from typing import Union

import numpy as np
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
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
    SUPPORTED_ROBOTS = ["panda_peg_insertion"]
    agent: Union[PandaForPegInsertion]
    _clearance = 0.005

    def __init__(
            self,
            *args,
            robot_uids="panda_peg_insertion",
            **kwargs,
    ):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.4, -0.4, 0.4], [0.05, -0.1, 0.2])
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]
