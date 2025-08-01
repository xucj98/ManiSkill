import argparse

from omegaconf import DictConfig, OmegaConf

from odpc.evaluation.rollout_analysis import analyze_rollout
from odpc.utils.utils import load_config_with_defaults, parse_config_expr


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, unknown = parser.parse_known_args()

    cfg = load_config_with_defaults(args.config)
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(cfg, cli)

    print("======== Analyze Rollout Config ========")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("======== Analyze Rollout Config ========")
 
    return args, cfg


if __name__ == "__main__":
    args, cfg = get_args()
    analyze_rollout(cfg)
