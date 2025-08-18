import wandb
import pandas as pd
import numpy as np

# --- 配置 ---
# 替换为你的实体（用户名或团队名）和项目名
ENTITY = None
PROJECT = "ManiSkill"
# 你想要处理的指标的名称
METRIC_NAME = "eval/env_ind/success_once"
# 我们将要创建的新指标的名称
NEW_METRIC_NAME = "eval/env_ind/success_once_max_3_mean"
# 取最大的N个值
N = 3

# --- 初始化 W&B API ---
api = wandb.Api()

# --- 获取项目中的所有实验 ---
try:
    runs = api.runs(f"{PROJECT}")
except Exception as e:
    print(f"获取项目 '{ENTITY}/{PROJECT}' 失败，请检查实体和项目名称是否正确。错误信息: {e}")
    exit()

print(f"找到了 {len(runs)} 个实验，开始处理...")

# --- 遍历所有实验 ---
for run in runs:
    print(f"正在处理实验: {run.name} ({run.id})")
    
    # 检查这个新指标是否已经计算过，如果计算过就跳过
    if NEW_METRIC_NAME in run.summary:
        print(f" -> 已跳过，因为 '{NEW_METRIC_NAME}' 已存在。")
        continue

    # 【快速检查】在调用 API 获取详细历史前，先检查 summary 中是否存在该指标
    # 这是一个很好的实践，可以避免对不相关的实验发起不必要的请求
    if METRIC_NAME not in run.summary:
        print(f" -> 已跳过，因为指标 '{METRIC_NAME}' 不存在于此实验的 summary 中。")
        continue

    try:
        # --- 核心改动：使用 history() ---
        # 直接获取一个 Pandas DataFrame，这对于小数据量非常方便
        history_df = run.history(keys=[METRIC_NAME])

        # 检查 DataFrame 是否为空或不包含我们需要的列
        if history_df.empty or METRIC_NAME not in history_df.columns:
            print(f" -> 警告：虽然 summary 中有记录，但在详细历史中找不到指标 '{METRIC_NAME}'。")
            continue

        # 从 DataFrame 中提取指标列，并用 .dropna() 移除任何缺失值 (NaN)
        metric_values = history_df[METRIC_NAME].dropna()

        # 检查有效数据点的数量是否足够
        if len(metric_values) < N:
            print(f" -> 警告：指标 '{METRIC_NAME}' 的有效数据点 ({len(metric_values)}个) 少于所需的 {N} 个，无法计算。")
            continue
            
        # --- 直接使用 Pandas 的功能进行计算 ---
        # 找出最大的 N 个值，然后计算它们的均值
        top_n_mean = metric_values.nlargest(N).mean()

        # --- 更新实验的摘要（Summary） ---
        run.summary[NEW_METRIC_NAME] = top_n_mean
        run.summary.update()
        
        print(f" -> 成功！计算出的 '{NEW_METRIC_NAME}' 为: {top_n_mean:.4f}")

    except Exception as e:
        print(f" -> 处理实验 {run.name} 时发生严重错误: {e}")

print("\n所有实验处理完毕！")