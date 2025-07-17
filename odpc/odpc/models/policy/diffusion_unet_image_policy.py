import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from odpc.utils.utils import instantiate_from_config
from odpc.models.vision import BaseVisionEncoder
from odpc.models.state import BaseStateEncoder
from odpc.models.policy.base_policy import BasePolicy
from odpc.models.modules.normalizer import BaseNormalizer

class DiffusionUnetImagePolicy(BasePolicy):
    def __init__(
            self, 
            obs_horizon,
            pred_horizon,    
            output_dim,
            noise_pred_net_config,
            noise_scheduler_config,
            action_normalizer_config=None,
            visual_encoder_config=None,
            state_encoder_config=None,
    ):
        super().__init__(
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            output_dim=output_dim,
        )
        
        self.visual_encoder: Optional[BaseVisionEncoder] = None
        self.state_encoder: Optional[BaseStateEncoder] = None
        if visual_encoder_config is not None:
            self.visual_encoder = instantiate_from_config(visual_encoder_config)
        if state_encoder_config is not None:
            self.state_encoder = instantiate_from_config(state_encoder_config)

        self.noise_pred_net: ConditionalUnet1D = instantiate_from_config(noise_pred_net_config)
        self.noise_scheduler: DDPMScheduler = instantiate_from_config(noise_scheduler_config)
        self.action_normalizer: BaseNormalizer = instantiate_from_config(action_normalizer_config)

    def compute_loss(
            self, 
            obs: dict, 
            action: torch.Tensor,
    ) -> torch.Tensor:
        action = self.action_normalizer(action)
       
        B = action.shape[0]
        device = action.device

        conds = []
        if self.visual_encoder is not None:
            conds.append(self.visual_encoder(obs))
        if self.state_encoder is not None:
            conds.append(self.state_encoder(obs))
        obs_cond = torch.cat(conds, dim=-1)  # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.output_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(action, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            sample=noisy_action_seq,
            timestep=timesteps,
            global_cond=obs_cond,
        )

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(action, noise, timesteps)
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = action
        else:
            raise TypeError("prediction type not recognized.")

        return F.mse_loss(noise_pred, target)

    def get_action(
            self, 
            obs: dict,
    ) -> torch.Tensor:
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq['state']: (B, obs_horizon, obs_state_dim)
        with torch.no_grad():
            conds = []
            if self.visual_encoder is not None:
                conds.append(self.visual_encoder(obs))
            if self.state_encoder is not None:
                conds.append(self.state_encoder(obs))
            obs_cond = torch.cat(conds, dim=-1)  # (B, obs_horizon * obs_dim)

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

        noisy_action_seq = self.action_normalizer.unnormalize(noisy_action_seq)

        return noisy_action_seq  # (B, act_horizon, output_dim)