import os
import argparse
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf
import json
import time

from odpc.data.demo.motionplanning import _main as motion_planning_single
from odpc.data.demo.motionplanning import MP_SOLUTIONS


def get_args():
    parser = argparse.ArgumentParser(description="分析专家策略在不同任务难度下的局限性")
    parser.add_argument("--config", type=str, required=True, 
                       help="演示配置文件路径 (odpc/configs/demo/下的yaml文件)")
    parser.add_argument("--difficulty_key", type=str, required=True,
                       help="难度参数在env_kwargs中的键名 (如 'clearance')")
    parser.add_argument("--difficulty_values", type=float, nargs='+', required=True,
                       help="难度参数的值列表 (如 0.02 0.03 0.04 0.05 0.06 0.07 0.10)")
    parser.add_argument("--num_trials", type=int, default=100,
                       help="每个难度参数下的测试次数 (默认: 100)")
    parser.add_argument("--output_dir", type=str, default="expert_analysis",
                       help="输出目录 (默认: expert_analysis)")
    args = parser.parse_args()
    return args


def test_expert_at_difficulty(cfg, difficulty_key, difficulty_value, num_trials, proc_id=0):
    """在指定难度下测试专家策略"""
    # 创建临时配置
    test_cfg = OmegaConf.create(cfg)
    test_cfg.motion_planning_args.env_kwargs[difficulty_key] = difficulty_value
    test_cfg.motion_planning_args.num_traj = num_trials
    test_cfg.motion_planning_args.only_count_success = False
    test_cfg.motion_planning_args.record_dir = f"temp_analysis_{proc_id}"
    test_cfg.motion_planning_args.traj_name = f"difficulty_{difficulty_value}_proc_{proc_id}"
    test_cfg.motion_planning_args.save_video = False
    test_cfg.motion_planning_args.vis = False
    
    # 确保临时目录存在
    os.makedirs(test_cfg.motion_planning_args.record_dir, exist_ok=True)
    
    print(f"开始测试难度 {difficulty_value} (进程 {proc_id})")
    start_time = time.time()
    
    # 运行专家策略测试
    output_path = motion_planning_single(test_cfg.motion_planning_args, proc_id, 0)
    
    elapsed_time = time.time() - start_time
    print(f"难度 {difficulty_value} 测试完成，耗时 {elapsed_time:.1f}秒")
    
    # 读取结果
    json_path = output_path.replace(".h5", ".json")
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # 统计成功率
    successes = []
    episode_lengths = []
    for episode in results['episodes']:
        success = episode.get('success', False)
        successes.append(success)
        if 'episode_length' in episode:
            episode_lengths.append(episode['episode_length'])
    
    success_rate = np.mean(successes)
    avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0
    
    # 清理临时文件
    if output_path and os.path.exists(output_path):
        os.remove(output_path)
    if json_path and os.path.exists(json_path):
        os.remove(json_path)
    if os.path.exists(test_cfg.motion_planning_args.record_dir):
        import shutil
        shutil.rmtree(test_cfg.motion_planning_args.record_dir)

    return {
        'difficulty_value': difficulty_value,
        'success_rate': success_rate,
        'num_successes': sum(successes),
        'num_trials': len(successes),
        'avg_episode_length': avg_episode_length,
        'successes': successes,
        'elapsed_time': elapsed_time
    }


def analyze_expert_limitations(args):
    """分析专家策略局限性"""
    # 加载配置
    cfg = OmegaConf.load(args.config)
    
    # 验证环境是否支持
    env_id = cfg.env_id
    if env_id not in MP_SOLUTIONS:
        raise RuntimeError(f"环境 {env_id} 没有可用的专家策略。可用选项: {list(MP_SOLUTIONS.keys())}")
    
    # 验证难度参数是否存在
    if args.difficulty_key not in cfg.motion_planning_args.env_kwargs:
        raise RuntimeError(f"难度参数 '{args.difficulty_key}' 在环境配置中不存在")
    
    print(f"开始分析专家策略局限性...")
    print(f"环境: {env_id}")
    print(f"难度参数: {args.difficulty_key}")
    print(f"难度值: {args.difficulty_values}")
    print(f"每个难度测试次数: {args.num_trials}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 并行测试不同难度
    results = []
    difficulty_values = args.difficulty_values
    pool = mp.Pool(len(difficulty_values))
    tasks = []
    for i, difficulty_value in enumerate(difficulty_values):
        proc_id = i
        task = pool.apply_async(
            test_expert_at_difficulty,
            (cfg, args.difficulty_key, difficulty_value, args.num_trials, proc_id)
        )
        tasks.append(task)
    
    for i, task in enumerate(tqdm(tasks, desc="测试不同难度")):
        result = task.get()
        results.append(result)
    pool.close()
    pool.join()
    
    # 按难度值排序
    results.sort(key=lambda x: x['difficulty_value'])
    
    # 保存结果
    output_file = os.path.join(args.output_dir, f"expert_analysis_{args.difficulty_key}.json")
    with open(output_file, 'w') as f:
        json.dump({
            'env_id': env_id,
            'difficulty_key': args.difficulty_key,
            'difficulty_values': args.difficulty_values,
            'num_trials': args.num_trials,
            'results': results
        }, f, indent=2)
    
    # 打印结果
    print("\n专家策略局限性分析结果:")
    print("=" * 80)
    print(f"{'难度值':<10} {'成功率':<10} {'成功数/总数':<15} {'平均步数':<10} {'耗时(秒)':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['difficulty_value']:<10.3f} "
              f"{result['success_rate']:<10.3f} "
              f"{result['num_successes']}/{result['num_trials']:<15} "
              f"{result['avg_episode_length']:<10.1f} "
              f"{result.get('elapsed_time', 0):<10.1f}")
    
    # 绘制成功率曲线
    plot_success_rate_curve(results, args.difficulty_key, env_id, args.output_dir)
    
    return results


def plot_success_rate_curve(results, difficulty_key, env_id, output_dir):
    """绘制成功率曲线"""
    difficulty_values = [r['difficulty_value'] for r in results]
    success_rates = [r['success_rate'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(difficulty_values, success_rates, 'bo-', linewidth=2, markersize=8)
    plt.xlabel(f'Task Difficulty ({difficulty_key})', fontsize=12)
    plt.ylabel('Expert Policy Success Rate', fontsize=12)
    plt.title(f'{env_id} Expert Policy Success Rate vs Task Difficulty', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    # 添加数据点标注
    for i, (x, y) in enumerate(zip(difficulty_values, success_rates)):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    # 保存图片
    plot_file = os.path.join(output_dir, f"success_rate_curve_{difficulty_key}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"成功率曲线已保存到: {plot_file}")


def main():
    args = get_args()
    
    # 设置多进程启动方法
    mp.set_start_method("spawn", force=True)
    
    # 分析专家策略局限性
    results = analyze_expert_limitations(args)
    
    print(f"\n分析完成！结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main() 