import os
import argparse # 用于处理命令行参数，如配置文件路径
from omegaconf import OmegaConf, DictConfig

from odpc.utils.utils import instantiate_from_config
from odpc.evaluation.precision_curse.precision_curse_analyzer import PrecisionCurseAnalyzer # 确保路径正确


def get_args():
    parser = argparse.ArgumentParser(description="运行精度诅咒分析流程")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/analysis/PegIns_EeDeltaPose_HoleSize.yaml", # 默认配置文件路径
        help="指向分析配置文件的路径 (相对于项目根目录)",
    )

    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(cfg, cli)

    return args, cfg


if __name__ == "__main__":
    args, cfg = get_args()

    os.makedirs(cfg.analyzer.save_dir, exist_ok=True)    
    OmegaConf.save(cfg, f"{cfg.analyzer.save_dir}/config.yaml", resolve=True)
    print(f"save_dir: {os.path.abspath(cfg.analyzer.save_dir)}")

    analyzer: PrecisionCurseAnalyzer = instantiate_from_config(cfg.analyzer)
    analyzer.run()
