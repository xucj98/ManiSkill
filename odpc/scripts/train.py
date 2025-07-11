import argparse
from omegaconf import OmegaConf

from odpc.training.train import train
from odpc.utils.utils import instantiate_from_config, parse_config_expr, load_config_with_defaults


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/odpc/base.yaml")
    args, unknown = parser.parse_known_args()

    cfg = load_config_with_defaults(args.config)
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(cfg, cli)

    return args, cfg
 

if __name__ == "__main__":
    args, cfg = get_args()
    train(cfg)
    