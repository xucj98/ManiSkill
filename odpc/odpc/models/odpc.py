import torch
import torch.nn as nn
import torch.nn.functional as F

from odpc.utils.utils import instantiate_from_config
from odpc.models.vision import BaseVisionEncoder

class ODPCModel(nn.Module):
    def __init__(
            self, 
            obs_horizon,
            pred_horizon,    
            output_dim,
            visual_encoder_config,
            noise_pred_net_config,
            noise_scheduler_config,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.output_dim = output_dim
        self.visual_encoder: BaseVisionEncoder = instantiate_from_config(visual_encoder_config)
        self.noise_pred_net = instantiate_from_config(noise_pred_net_config)
        self.noise_scheduler = instantiate_from_config(noise_scheduler_config)

    def compute_loss(self, obs_seq, action_seq):
        B = action_seq.shape[0]
        device = action_seq.device

        # observation as FiLM conditioning
        obs_cond = self.visual_encoder(obs_seq)  # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.output_dim), device=device)

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
        with torch.no_grad():
            obs_cond = self.visual_encoder(obs_seq)  # (B, obs_horizon * obs_dim)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn(
                (obs_cond.shape[0], self.pred_horizon, self.output_dim), device=obs_cond.device
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

        return noisy_action_seq  # (B, act_horizon, output_dim)