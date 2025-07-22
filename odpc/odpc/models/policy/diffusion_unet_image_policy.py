import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange

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
        self.action_normalizer: Optional[BaseNormalizer] = None
        if action_normalizer_config is not None:
            self.action_normalizer = instantiate_from_config(action_normalizer_config)

    def compute_loss(
            self, 
            obs: dict, 
            action: torch.Tensor,
    ) -> torch.Tensor:
        if self.action_normalizer is not None:
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

            if self.action_normalizer is not None:
                noisy_action_seq = self.action_normalizer.unnormalize(noisy_action_seq)

        return noisy_action_seq  # (B, act_horizon, output_dim)
    
    @torch.no_grad()
    def compute_avg_loss(
            self, 
            obs: dict,
            action: torch.Tensor,
            n_timesteps: int = 4,
    ) -> torch.Tensor:
        """
        对于每个 (obs, act) 样本，计算 n * timesteps 个 diffusion loss, 然后取平均
        """
        B = action.shape[0]
        device = action.device
        
        conds = []
        if self.visual_encoder is not None:
            conds.append(self.visual_encoder(obs))
        if self.state_encoder is not None:
            conds.append(self.state_encoder(obs))
        obs_cond = torch.cat(conds, dim=-1)  # (B, obs_horizon * obs_dim)

        # 重复 obs_cond 和 action 到 n * timesteps 个
        n = n_timesteps * len(self.noise_scheduler.timesteps)
        obs_cond = repeat(obs_cond, 'b d -> (b n) d', n=n)
        action = repeat(action, 'b t d -> (b n) t d', n=n)

        # sample noise to add to actions
        noise = torch.randn((B * n, self.pred_horizon, self.output_dim), device=device)

        timesteps = repeat(self.noise_scheduler.timesteps, 'k -> (b n k)', 
                           b=B, n=n_timesteps).to(device)

        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)

        noise_pred = self.noise_pred_net(
            sample=noisy_action,
            timestep=timesteps,
            global_cond=obs_cond,
        )

        loss = F.mse_loss(noise_pred, noise, reduction='none')
        loss = rearrange(loss, '(b n) t d -> b (n t d)', b=B, n=n)
        loss = loss.mean(dim=-1)
        return loss