from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from mani_skill.utils import common

def evaluate(n: int, agent, eval_envs, device, sim_backend: str, progress_bar: bool = True):
    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n)
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        while eps_count < n:
            obs = common.to_tensor(obs, device)
            action_seq = agent.get_action(obs)
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()
            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                if truncated.any():
                    break

            if truncated.any():
                assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                eps_count += eval_envs.num_envs
                if progress_bar:
                    pbar.update(eval_envs.num_envs)
    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics

def evaluate_on_dataset(dataset, agent, args, device):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    se = 0.
    n = 0.
    pbar = tqdm(total=len(dataloader))
    for batch in dataloader:
        batch = common.to_tensor(batch, device)
        observations = batch["observations"]
        observations["rgb"] = observations["rgb"].permute(0, 1, 3, 4, 2)
        observations["depth"] = observations["depth"].permute(0, 1, 3, 4, 2)
        action = agent.get_action(observations)
        action_gt = batch["actions"][:, args.obs_horizon - 1: args.obs_horizon + args.act_horizon - 1]
        se += (action - action_gt).pow(2).sum()
        n += action_gt.numel()
        pbar.update(1)
        pbar.set_postfix({"mse": (se / n).item()})
    pbar.close()
    return {"mse": (se / n).item()}


def evaluate_odpc(n: int, agent, eval_envs, device, sim_backend: str, progress_bar: bool = True):
    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n)
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        agent.reset()
        eps_count = 0
        while eps_count < n:
            obs = common.to_tensor(obs, device)
            action_seq = agent.get_action(obs)
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()
            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                if truncated.any():
                    break
            if truncated.any():
                agent.reset()
                assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                eps_count += eval_envs.num_envs
                if progress_bar:
                    pbar.update(eval_envs.num_envs)
    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics
