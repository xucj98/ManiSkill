import cv2
import numpy as np
import sapien
import trimesh.primitives

ALGO_NAME = "BC_Diffusion_rgbd_UNet"

from gymnasium.spaces import Dict, Box
from gymnasium.vector.vector_env import VectorEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_invert, quaternion_multiply, quaternion_to_matrix, matrix_to_euler_angles)
from mani_skill.utils import common
from mani_skill.utils.structs.pose import Pose
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb

from diffusion_policy.data_converison import DataConversion, pose_multiply, pose_inv
from diffusion_policy.conditional_unet1d import ConditionalUnet1D

from torchvision.models.resnet import resnet18


class RGBDAgent(nn.Module):
    def __init__(self, env: VectorEnv, args):
        super().__init__()
        self.args = args
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert (
                len(env.single_observation_space["state"].shape) == 2
        )  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (
                env.single_action_space.low == -1
        ).all()
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = env.single_observation_space["state"].shape[1]
        total_visual_channels = 0
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()

        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        visual_feature_dim = 256
        # self.visual_encoder = PlainConv(
        #     in_channels=total_visual_channels, out_dim=visual_feature_dim, pool_feature_map=True
        # )
        self.visual_encoder = resnet18(num_classes=visual_feature_dim)
        if args.used_cameras != "all":
            num_cameras = len(args.used_cameras.split(","))
            total_visual_channels = self.include_rgb * 3 * num_cameras + self.include_depth * num_cameras
        self.visual_encoder.conv1 = nn.Conv2d(total_visual_channels, 64,
                                              kernel_size=7, stride=2, padding=3, bias=False)

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=self.obs_horizon * (visual_feature_dim + (obs_state_dim if args.use_state else 0)),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type="epsilon",  # predict noise (instead of denoised action)
        )

    def encode_obs(self, obs_seq, eval_mode):
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3*k, H, W)
            if self.args.used_cameras != "all":
                camera_ids = [int(x) for x in self.args.used_cameras.split(",")]
                channel_ids = [x for k in camera_ids for x in range(k * 3, k * 3 + 3)]
                rgb = rgb[:, :, channel_ids]
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float()  # (B, obs_horizon, 1*k, H, W)
            if self.args.depth_clamp > 0:
                depth[depth < 0] = 0
                depth[depth > self.args.depth_clamp] = 0
            if self.args.used_cameras != "all":
                camera_ids = [int(x) for x in self.args.used_cameras.split(",")]
                depth = depth[:, :, camera_ids]
            depth = depth / 1024.0
            img_seq = depth
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W), C=4*k
        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)  # (B*obs_horizon, C, H, W)
        visual_feature = self.visual_encoder(img_seq)  # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(
            batch_size, self.obs_horizon, visual_feature.shape[1]
        )  # (B, obs_horizon, D)
        if self.args.use_state:
            feature = torch.cat(
                (visual_feature, obs_seq["state"]), dim=-1
            )  # (B, obs_horizon, D+obs_state_dim)
        else:
            feature = visual_feature
        return feature.flatten(start_dim=1)  # (B, obs_horizon * (D+obs_state_dim))

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq["state"].shape[0]
        device = obs_seq["state"].device

        # observation as FiLM conditioning
        obs_cond = self.encode_obs(
            obs_seq, eval_mode=False
        )  # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond
        )

        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq['state']: (B, obs_horizon, obs_state_dim)
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            obs_cond = self.encode_obs(
                obs_seq, eval_mode=True
            )  # (B, obs_horizon * obs_dim)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)


class ODPCAgent(nn.Module):
    def __init__(self, env: VectorEnv, args, act_dim):
        super().__init__()
        self.args = args
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        self.act_dim = act_dim
        obs_state_dim = env.single_observation_space["state"].shape[1]
        total_visual_channels = 0
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()

        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        visual_feature_dim = 256
        self.visual_encoder = resnet18(num_classes=visual_feature_dim)
        if args.used_cameras != "all":
            num_cameras = len(args.used_cameras.split(","))
            total_visual_channels = self.include_rgb * 3 * num_cameras + self.include_depth * num_cameras
        self.visual_encoder.conv1 = nn.Conv2d(total_visual_channels, 64,
                                              kernel_size=7, stride=2, padding=3, bias=False)

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=self.obs_horizon * (visual_feature_dim + (obs_state_dim if args.use_state else 0)),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type="epsilon",  # predict noise (instead of denoised action)
        )

    def encode_obs(self, obs_seq, eval_mode):
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3*k, H, W)
            if self.args.used_cameras != "all":
                camera_ids = [int(x) for x in self.args.used_cameras.split(",")]
                channel_ids = [x for k in camera_ids for x in range(k * 3, k * 3 + 3)]
                rgb = rgb[:, :, channel_ids]
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float()  # (B, obs_horizon, 1*k, H, W)
            if self.args.depth_clamp > 0:
                depth[depth < 0] = 0
                depth[depth > self.args.depth_clamp] = 0
            if self.args.used_cameras != "all":
                camera_ids = [int(x) for x in self.args.used_cameras.split(",")]
                depth = depth[:, :, camera_ids]
            depth = depth / 1024.0
            img_seq = depth
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W), C=4*k
        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)  # (B*obs_horizon, C, H, W)
        visual_feature = self.visual_encoder(img_seq)  # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(
            batch_size, self.obs_horizon, visual_feature.shape[1]
        )  # (B, obs_horizon, D)
        if self.args.use_state:
            feature = torch.cat(
                (visual_feature, obs_seq["state"]), dim=-1
            )  # (B, obs_horizon, D+obs_state_dim)
        else:
            feature = visual_feature
        return feature.flatten(start_dim=1)  # (B, obs_horizon * (D+obs_state_dim))

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq["rgb"].shape[0]
        device = obs_seq["rgb"].device

        # observation as FiLM conditioning
        obs_cond = self.encode_obs(
            obs_seq, eval_mode=False
        )  # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond
        )

        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq, clip=True):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq['state']: (B, obs_horizon, obs_state_dim)
        B = obs_seq["rgb"].shape[0]
        with torch.no_grad():
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            obs_cond = self.encode_obs(
                obs_seq, eval_mode=True
            )  # (B, obs_horizon * obs_dim)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq["rgb"].device
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        if clip:
            noisy_action_seq = noisy_action_seq[:, start:end]
        return noisy_action_seq  # (B, act_horizon, act_dim)


class ODPCAgentWrapper(nn.Module):
    def __init__(
            self,
            agent: ODPCAgent,
            envs,
            dc: DataConversion,
            origin_obs_space,
            robot_uid,
            video_dir=None,
    ):
        super(ODPCAgentWrapper, self).__init__()
        self.agent: ODPCAgent = agent
        self.envs = envs
        self.dc = dc
        self.num_envs = envs.num_envs
        self.origin_obs_space = origin_obs_space
        self.act_horizon = agent.act_horizon
        self.control_mode = dc.control_mode
        self.video_dir = video_dir

        self.stages = torch.zeros(envs.num_envs)
        self.action_step = 0
        self.agent_action = torch.zeros((envs.num_envs, agent.pred_horizon, dc.control_dim))

        self.grasp_pose = torch.zeros((envs.num_envs, 7))
        self.reach_pose = torch.zeros((envs.num_envs, 7))
        if "panda" in robot_uid:
            self.gripper_state = torch.tensor([1., -1.])  # open, close
        elif "robotiq" in robot_uid:
            self.gripper_state = torch.tensor([0., 0.81])  # open, close
        else:
            raise NotImplementedError

    def state_to_dict(self, state, ref_dict):
        state_dict = {}
        for k, v in ref_dict.items():
            if k in ['sensor_data', 'sensor_param']:
                continue
            if isinstance(v, Dict):
                state, state_dict[k] = self.state_to_dict(state, v)
            elif isinstance(v, Box):
                state_dict[k] = state[..., :v.shape[-1]]
                state = state[..., v.shape[-1]:]
        return state, state_dict

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose ()."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    def get_grasp_pose(
            self,
            peg_half_size,
            peg_pose,
            ee_pose
    ):
        for i in range(self.num_envs):
            extents = peg_half_size[i].cpu().numpy() * 2
            transform = (
                    Pose.create(peg_pose[i]) * sapien.Pose([-0.07, 0, 0])
            ).to_transformation_matrix()[0].cpu().numpy()
            obb = trimesh.primitives.Box(extents=extents, transform=transform)
            approaching = np.array([0, 0, -1])
            target_closing = Pose.create(ee_pose[i]).to_transformation_matrix()[0, :3, 1].cpu().numpy()
            grasp_info = compute_grasp_info_by_obb(
                obb, approaching=approaching, target_closing=target_closing, depth=0.025)
            closing, center = grasp_info["closing"], grasp_info["center"]
            grasp_pose = self.build_grasp_pose(approaching, closing, center)
            self.grasp_pose[i, :3] = torch.from_numpy(grasp_pose.p)
            self.grasp_pose[i, 3:] = torch.from_numpy(grasp_pose.q)
            reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
            self.reach_pose[i, :3] = torch.from_numpy(reach_pose.p)
            self.reach_pose[i, 3:] = torch.from_numpy(reach_pose.q)

    @staticmethod
    def make_grid(images: np.ndarray) -> np.ndarray:
        n = len(images)
        nh = int(np.sqrt(n))
        nw = int(np.ceil(n / nh))
        h, w = images.shape[1:3]
        grid = np.zeros((nh * h, nw * w, 3), np.uint8)
        for i, rgb in enumerate(images):
            x, y = (i % nh) * h, (i // nh) * w
            grid[x: x + h, y: y + w] = rgb
        return grid

    def get_action(self, obs_seq):
        _, state_dict = self.state_to_dict(obs_seq["state"], self.origin_obs_space)
        ee_pose = state_dict["extra"]["tcp_pose"][:, 0, :]
        base_pose = state_dict["extra"]["base_pose"][:, 0, :]

        if self.action_step % self.act_horizon == 0:
            pred = self.agent.get_action(obs_seq, clip=False)
            self.agent_action = self.dc.pred_to_control(
                pred=pred.clone().detach(),
                poses_ee_cur=state_dict["extra"]["tcp_pose"][..., :1, :],
                poses_base=state_dict["extra"]["base_pose"][..., :1, :],
                poses_camera_world=state_dict["extra"]["cam0_world_pose"][..., :1, :],
            )

            if self.video_dir is not None:
                images = self.dc.pred_to_visualize(
                    rgb=obs_seq["rgb"][..., :3, :, :],
                    pred=pred.clone().detach(),
                    poses_cam_obj_cur=state_dict["extra"]["cam0_peg_pose"][..., :1, :],
                )
                grid = self.make_grid(images)
                cv2.imwrite(f"{self.video_dir}/{self.action_step:04d}.jpg", grid[:, :, ::-1])

        if self.stages[0] == 0:
            self.get_grasp_pose(
                peg_half_size=state_dict['extra']['peg_half_size'][:, -1, :],
                peg_pose=state_dict['extra']['peg_pose'][:, -1, :],
                ee_pose=ee_pose,
            )
            self.stages[:] = 1

        # get mp_action (motion-planning action)
        def reach_target(threshold):
            dp = torch.sum(torch.abs(ee_pose[i, :3] - mp_target_pose[i, :3])).item()
            dq1 = torch.sum(torch.abs(ee_pose[i, 3:] - mp_target_pose[i, 3:])).item()
            dq2 = torch.sum(torch.abs(ee_pose[i, 3:] + mp_target_pose[i, 3:])).item()
            return dp + min(dq1, dq2) < threshold

        mp_target_pose = torch.zeros_like(ee_pose)

        for i in range(self.num_envs):
            if self.stages[i] == 1:
                mp_target_pose[i] = self.reach_pose[i]
                if reach_target(0.05):
                    self.stages[i] = 2
            if self.stages[i] == 2:
                mp_target_pose[i] = self.grasp_pose[i]
                if reach_target(0.02):
                    self.stages[i] = 3
            if 3 <= self.stages[i] < 4:
                mp_target_pose[i] = self.grasp_pose[i]
                self.stages[i] += 1 / 20
            if 4 <= self.stages[i] < 5:
                mp_target_pose[i] = self.reach_pose[i]
                if reach_target(0.05):
                    self.stages[i] = 5
        base_target = pose_multiply(pose_inv(base_pose), mp_target_pose)
        if self.control_mode == "pd_ee_delta_pose":
            base_ee = pose_multiply(pose_inv(base_pose), ee_pose)
            p = base_target[..., :3] - base_ee[..., :3]
            q = quaternion_multiply(base_target[..., 3:], quaternion_invert(base_ee[..., 3:]))
            euler = matrix_to_euler_angles(quaternion_to_matrix(q), "XYZ")
            mp_action = torch.cat([p, -euler], dim=-1)
        elif self.control_mode == "pd_ee_pose":
            p = base_target[..., :3]
            q = base_target[..., 3:]
            euler = matrix_to_euler_angles(quaternion_to_matrix(q), "XYZ")
            mp_action = torch.cat([p, euler], dim=-1)
        else:
            raise NotImplementedError

        # use agent action
        for i in range(self.num_envs):
            if self.stages[i] == 5:
                mp_action[i] = self.agent_action[i][self.action_step % self.act_horizon]

        gripper = self.gripper_state[(self.stages >= 3).int()].to(mp_action)
        action = torch.cat([mp_action, gripper[:, None]], dim=-1)

        self.action_step += 1

        return action[:, None, :]

    def reset(self):
        self.stages[:] = 0
        self.action_step = 0
