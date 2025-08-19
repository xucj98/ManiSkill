import wandb
import pandas as pd
import numpy as np

PROJECT = "ManiSkill"

api = wandb.Api()
runs = api.runs(
    f"{PROJECT}",
    filters={
        "group": "RandomFramesLossFinder",
    }
)

for idx, run in enumerate(runs):
    print(f"正在处理实验({idx+1}/{len(runs)}): {run.name} {run.id}")

    if "data_trajs_dict" in run.config:
        print(f" -> 已跳过，因为 'data_trajs_dict' 已存在。")
        continue

    if "data_paths" not in run.config:
        print(f" -> 已跳过，因为 'data_paths' 不存在。")
        continue

    data_paths = run.config["data_paths"]
    num_trajs = run.config["train_dataset"]["num_traj"]

    if not isinstance(num_trajs, dict):
        print(f" -> 已跳过，因为 'num_trajs' 不是字典。")
        continue

    data_trajs_dict = {}
    for data_id, data_path in enumerate(data_paths):
        data_trajs_dict[str(data_id)] = num_trajs[data_path.split("/")[-1]]

    run.config["data_trajs_dict"] = data_trajs_dict
    run.update()

    print(f" -> 已更新 'data_trajs_dict' 为 {data_trajs_dict}。")

