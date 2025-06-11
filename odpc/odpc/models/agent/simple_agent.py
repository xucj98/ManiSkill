import torch

from odpc.models.agent.base_agent import BaseAgent
from odpc.models.policy.base_policy import BasePolicy
from omegaconf import ListConfig


class SimpleAgent(BaseAgent):
    def __init__(self,
            model: BasePolicy,
            num_envs: int,
            act_horizon: int,
            pred_horizon: int,
            video_dir: str = None,
            obs_processor_configs: ListConfig = [],
    ):
        super().__init__(
            model=model,
            num_envs=num_envs,
            act_horizon=act_horizon,
            pred_horizon=pred_horizon,
            video_dir=video_dir,
            obs_processor_configs=obs_processor_configs,
        )

        self.model_action = torch.zeros((num_envs, pred_horizon, self.model.output_dim))


    def get_action(self, obs: dict, channel_last: bool = False) -> torch.Tensor:
        action_step = self._action_step % self.act_horizon
        if action_step == 0:
            if not channel_last:
                obs = self.permute_obs(obs, [0, 1, 3, 4, 2])
            for processor in self.obs_processors:
                obs = processor.process(obs)
            obs = self.permute_obs(obs, [0, 1, 4, 2, 3])
            self.model_action = self.model.get_action(obs)

        self._action_step += 1
        return self.model_action[:, action_step: action_step+1, :]
    