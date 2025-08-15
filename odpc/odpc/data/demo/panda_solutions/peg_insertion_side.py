import torch
import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.tasks import PegInsertionSideEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

from odpc.data.demo.motionplanner import PandaArmMotionPlanningClipSolver
from odpc.envs import PegInsertionSideV2Env

def main():
    env: PegInsertionSideV2Env = gym.make(
        "PegInsertionSide-v2",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )
    for seed in range(100):
        res = solve(env, seed=seed, debug=False, vis=True)
        print(res[-1])
    env.close()


def solve(
        env, 
        seed: int = None, 
        debug: bool = False, 
        vis: bool = False, 
        start_after_grasp: bool = False,
        pre_insert_aug: float = 0.,
        refine_steps: int = 3,
):
    env.reset(seed=seed)
    start_step = 0
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
    env: PegInsertionSideV2Env = env.unwrapped
    FINGER_LENGTH = 0.025

    obb = get_actor_obb(env.peg)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    grasp_info = compute_grasp_info_by_obb(
        obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = max(0.05, env.peg_half_sizes[0, 0].item() / 2 + 0.01)
    # offset = np.random.uniform(0.05, env.peg_half_sizes[0, 0].item() - 0.01)
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
    if start_after_grasp:
        start_step = res[-1]['elapsed_steps'].item()

    # -------------------------------------------------------------------------- #
    # Align Peg
    # -------------------------------------------------------------------------- #

    # align the peg with the hole
    ee_cur_pose = reach_pose

    # coarse insert pose
    if pre_insert_aug > 0 and np.random.rand() < pre_insert_aug:
        offset = 0.02 + env.peg_half_sizes[0, 0].item()
        rand_p = np.random.uniform(low=[-.01, -.05, -.05], high=[.01, .05, .05])
        rand_q = np.random.uniform(low=[1, -.2, -.2, -.2], high=[1, .2, .2, .2])
        rand_q = rand_q / np.linalg.norm(rand_q)
        coarse_insert_pose = env.goal_pose * sapien.Pose([-offset, 0, 0]) * sapien.Pose(rand_p, rand_q)
        delta_pose = coarse_insert_pose * env.peg.pose.inv()
        ee_cur_pose = delta_pose * ee_cur_pose
        res = planner.move_to_pose_with_screw(ee_cur_pose)
        if res == -1: return res
        start_step = res[-1]['elapsed_steps'].item()
        

    # fine insert pose
    offset = 0.01 + env.peg_half_sizes[0, 0].item()
    fine_insert_pose = env.goal_pose * sapien.Pose([-offset, 0, 0])
    # refine the insertion pose
    for _ in range(refine_steps):
        delta_pose = fine_insert_pose * env.peg.pose.inv()
        ee_cur_pose = delta_pose * ee_cur_pose
        res = planner.move_to_pose_with_screw(ee_cur_pose)
        if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Insert
    # -------------------------------------------------------------------------- #
    delta_pose = env.goal_pose * sapien.Pose([0.03, 0, 0]) * env.peg.pose.inv()
    ee_cur_pose = delta_pose * ee_cur_pose
    res = planner.move_to_pose_with_screw(ee_cur_pose)
    if res == -1: return res

    planner.close()
    res[-1]['start_step'] = start_step
    return res

def detect_current_stage(env, env_idx: int = 0) -> str:
    """Detects stage based on the official is_grasping method."""
    ALIGN_THRESH_XY = 0.03
    LIFT_CLEARANCE_Z = 0.015
    is_grasped = env.agent.is_grasping(env.peg)[env_idx]
    if not is_grasped:
        return "INITIAL"
    peg_p = env.peg.pose.p[env_idx].squeeze(0)
    goal_p = env.goal_pose.p[env_idx].squeeze(0)
    initial_peg_z = env.peg_half_sizes[env_idx, 2]
    is_lifted = peg_p[2] > (initial_peg_z + LIFT_CLEARANCE_Z)
    if not is_lifted:
        return "LIFT"
    is_aligned_xy = torch.linalg.norm(peg_p[:2] - goal_p[:2]) < ALIGN_THRESH_XY
    if is_aligned_xy:
        return "INSERT"
    else:
        return "ALIGN"

def solve_from_stage(
        env, 
        seed: int = None, 
        debug: bool = False, 
        vis: bool = False,
        clip_steps: int = 100,
        refine_steps: int = 3,
):    
    planner = PandaArmMotionPlanningClipSolver(
        env, 
        debug=debug, 
        vis=vis, 
        base_pose=env.unwrapped.agent.robot.pose, 
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
        clip_steps=clip_steps,
    )

    env: PegInsertionSideV2Env = env.unwrapped
    FINGER_LENGTH = 0.025
    
    obb = get_actor_obb(env.peg)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info = compute_grasp_info_by_obb(obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH)
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = max(0.05, env.peg_half_sizes[0, 0].item() / 2 + 0.01)
    grasp_pose.p = (env.peg.pose * sapien.Pose([-offset, 0, 0])).p[0].cpu().numpy()

    current_stage = detect_current_stage(env)

    if current_stage != "INITIAL":
        planner.close_gripper()
    else:
        planner.open_gripper()

    if current_stage == "INITIAL":
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        res = planner.move_to_pose(reach_pose)
        if res == -1: return -1
        res = planner.move_to_pose(grasp_pose)
        if res == -1: return -1
        planner.close_gripper()
        current_stage = "LIFT"

    if current_stage == "LIFT":
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        res = planner.move_to_pose(reach_pose)
        if res == -1: return -1
        current_stage = "ALIGN"

    ee_cur_pose = env.agent.tcp.pose

    if current_stage == "ALIGN":
        offset = 0.01 + env.peg_half_sizes[0, 0].item()
        fine_insert_pose = env.goal_pose * sapien.Pose([-offset, 0, 0])
        for _ in range(refine_steps):
            delta_pose = fine_insert_pose * env.peg.pose.inv()
            ee_cur_pose = delta_pose * ee_cur_pose
            res = planner.move_to_pose(ee_cur_pose)
            if res == -1: return -1
        current_stage = "INSERT"

    if current_stage == "INSERT":
        delta_pose = env.goal_pose * env.peg.pose.inv()
        ee_cur_pose = delta_pose * ee_cur_pose
        res = planner.move_to_pose(ee_cur_pose)
        if res == -1: return -1

    planner.close()
    return res

if __name__ == "__main__":
    main()
