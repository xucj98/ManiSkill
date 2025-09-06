from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop.roll_ball import RollBallEnv

@register_env("RollBall-v2", max_episode_steps=80)
class RollBallV2Env(RollBallEnv):
    def __init__(self, *args, goal_radius=0.1, ball_radius=0.035, **kwargs):
        self.goal_radius = goal_radius
        self.ball_radius  = ball_radius
        super().__init__(*args, **kwargs)


