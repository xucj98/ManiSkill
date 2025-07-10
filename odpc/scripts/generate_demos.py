import argparse
from omegaconf import OmegaConf

from odpc.data.demo.generation import run_generation_workflow
from odpc.utils.utils import load_config_with_defaults


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/demo/peg-insertion.yaml")
    args, unknown = parser.parse_known_args()

    cfg = load_config_with_defaults(args.config)
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(cfg, cli)
 
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    return args, cfg


if __name__ == "__main__":
    args, cfg = get_args()
    
    # 调用封装好的核心工作流
    final_path = run_generation_workflow(cfg)

    print(f"Demonstration generation finished. Final compressed trajectory at: {final_path}")
