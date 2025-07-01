import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from odpc.envs import StackCuboidEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.wrappers.record import RecordEpisode

def solve(
        env: StackCuboidEnv, 
        seed=None, 
        debug=False, 
        vis=False,
        start_after_grasp=False,
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
    )
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    
    # 获取长方体的实际尺寸（对于单环境，取第一个环境的尺寸）
    cuboidA_half_size = env.cuboidA_half_sizes[0].cpu().numpy()  # 取第一个环境的尺寸
    cuboidB_half_size = env.cuboidB_half_sizes[0].cpu().numpy()  # 取第一个环境的尺寸
    
    obb = get_actor_obb(env.cuboidA)

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # Search a valid pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")

    # -------------------------------------------------------------------------- #
    # Reach - 根据长方体高度调整接近距离
    # -------------------------------------------------------------------------- #
    # 使用固定的接近距离，但稍微增加一些
    approach_distance = 0.08  # 8cm的接近距离
    reach_pose = grasp_pose * sapien.Pose([0, 0, -approach_distance])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    res = planner.close_gripper()
    if start_after_grasp:
        start_step = res[-1]['elapsed_steps'].item()

    # -------------------------------------------------------------------------- #
    # Lift - 根据长方体高度调整提升高度
    # -------------------------------------------------------------------------- #
    # 根据长方体高度动态调整提升高度
    total_height = cuboidA_half_size[2] + cuboidB_half_size[2] * 2
    # 确保提升高度足够避开底部长方体，但不要太高导致运动规划失败
    lift_height = max(0.15, min(0.25, total_height + 0.05))  # 15cm-25cm之间
    lift_pose = sapien.Pose([0, 0, lift_height]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)

    # -------------------------------------------------------------------------- #
    # Stack - 移动到目标位置并放置
    # -------------------------------------------------------------------------- #
    # 计算目标位置：长方体B顶部中心
    goal_pose = env.cuboidB.pose * sapien.Pose([0, 0, cuboidA_half_size[2] + cuboidB_half_size[2] + 0.02])
    
    # 移动到目标位置上方
    intermediate_pose = sapien.Pose(goal_pose.p.cpu().numpy()[0] + np.array([0, 0, 0.05]), lift_pose.q)
    planner.move_to_pose_with_screw(intermediate_pose)

    # 再下降到最终释放位置
    final_place_pose = sapien.Pose(goal_pose.p.cpu().numpy()[0], lift_pose.q)
    planner.move_to_pose_with_screw(final_place_pose, refine_steps=5)

    # 松开夹爪
    res = planner.open_gripper()

    # 松开后再向上移动一段，保证脱离物体，并等待物体稳定
    retreat_pose = sapien.Pose(goal_pose.p.cpu().numpy()[0] + np.array([0, 0, 0.04]), lift_pose.q)  # 向上8cm
    res = planner.move_to_pose_with_screw(retreat_pose, refine_steps=20)

    planner.close()
    res[-1]['start_step'] = start_step
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    env = gym.make("StackCuboid-v1", render_mode="rgb_array")
    if args.record:
        env = RecordEpisode(env, "stack_cuboid.mp4")

    solve(env, seed=args.seed, debug=args.debug, vis=args.vis)

if __name__ == "__main__":
    main() 