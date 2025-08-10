import mplib
import numpy as np
import sapien
import trimesh

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

class PandaArmMotionPlanningClipSolver(PandaArmMotionPlanningSolver):

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        clip_steps: int = 100,
    ):
        super().__init__(env, debug, vis, base_pose, visualize_target_grasp_pose, print_env_info, joint_vel_limits, joint_acc_limits)
        self.clip_steps = clip_steps

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        if n_step > self.clip_steps:
            return -1
        return super().follow_path(result, refine_steps)

    def move_to_pose(self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0):
        res = self.move_to_pose_with_screw(pose, dry_run, refine_steps)
        if res == -1:
            res = self.move_to_pose_with_RRTConnect(pose, dry_run, refine_steps)
        return res