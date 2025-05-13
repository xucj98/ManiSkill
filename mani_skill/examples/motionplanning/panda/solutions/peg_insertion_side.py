import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.tasks import PegInsertionSideEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)


def main():
    env: PegInsertionSideEnv = gym.make(
        "PegInsertionSide-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )
    for seed in range(100):
        res = solve(env, seed=seed, debug=False, vis=True)
        print(res[-1])
    env.close()


def solve(env: PegInsertionSideEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )
    env = env.unwrapped
    FINGER_LENGTH = 0.025

    obb = get_actor_obb(env.peg)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    grasp_info = compute_grasp_info_by_obb(
        obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = np.random.uniform(0.05, env.peg_half_sizes[0, 0].item() - 0.01)
    grasp_pose.p = (env.peg.pose * sapien.Pose([-offset, 0, 0])).p[0]

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * (sapien.Pose([0, 0, -0.05]))
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res
    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper()

    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1 or res[-1]['elapsed_steps'].item() > 350: return -1

    # -------------------------------------------------------------------------- #
    # Align Peg
    # -------------------------------------------------------------------------- #

    # align the peg with the hole
    offset = 0.01 + env.peg_half_sizes[0, 0].item()
    peg_insert_pose = env.goal_pose * sapien.Pose([-offset, 0, 0])
    cur_pose  = reach_pose
    # refine the insertion pose
    for _ in range(3):
        delta_pose = peg_insert_pose * env.peg.pose.inv()
        cur_pose = delta_pose * cur_pose
        res = planner.move_to_pose_with_screw(cur_pose)
        if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Insert
    # -------------------------------------------------------------------------- #
    delta_pose = env.goal_pose * sapien.Pose([0.03, 0, 0]) * env.peg.pose.inv()
    insert_pose = delta_pose * cur_pose
    res = planner.move_to_pose_with_screw(insert_pose)
    if res == -1: return res

    planner.close()
    return res


if __name__ == "__main__":
    main()
