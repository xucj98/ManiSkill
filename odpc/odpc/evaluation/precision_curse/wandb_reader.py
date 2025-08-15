import wandb
from typing import List, Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf # 用于类型提示和处理配置
from odpc.utils.utils import get_nested_value


class WandbReader:
    def __init__(self,
                 project: str,
                 filters: List[DictConfig], # 列表中的每个元素是一个筛选组配置
                 entity: Optional[str] = None,
                 **kwargs): # 捕获其他未明确定义的参数
        self.api = wandb.Api()
        self.entity = entity
        self.project = project
        self.filters_config = filters # OmegaConf ListConfig or list of DictConfig
        print(f"WandbReader initialized for entity='{entity}', project='{project}'. "
              f"Filters: {len(filters) if filters else 0}")

    def _matches_condition(self, run: wandb.apis.public.Run, condition: DictConfig) -> bool:
        """
        检查单个运行是否满足单个条件。
        condition 包含: key, value, op
        """
        key = condition.key
        expected_value = condition.value
        op = condition.get("op", "==") # 默认为等于操作

        actual_value = None
        if key.lower() == "group":
            actual_value = run.group
        elif key.lower() == "name":
            actual_value = run.name
        elif key.lower() == "id":
            actual_value = run.id
        elif key.lower() == "job type": # wandb 中的 Job Type
            actual_value = run.job_type
        elif key.lower() == "state":
            actual_value = run.state
        elif key.lower().startswith("config."): # 检查配置项
            try:
                actual_value = get_nested_value(run.config, key[7:])
            except KeyError:
                return False
        elif key.lower().startswith("summary."): # 检查摘要指标
            try:
                actual_value = get_nested_value(run.summary, key[8:])
            except KeyError:
                return False # 如果摘要路径不存在，则不匹配
        elif key.lower() == "tags": # 检查标签 (expected_value 应该是一个标签字符串)
            if op == "contains":
                return expected_value in run.tags
            else: # 默认为 "==" 检查，即标签列表完全匹配 (通常不这么用)
                # 为了简化，我们假设对于 "tags"，"==" 意味着 "contains"
                return expected_value in run.tags
        else:
            return False # 或 True，取决于严格程度

        if actual_value is None and expected_value is not None: # 如果无法从 run 中获取值，但期望有值
            return False

        # 执行操作比较
        if op == "==":
            return actual_value == expected_value
        elif op == "!=":
            return actual_value != expected_value
        elif op == ">":
            return actual_value > expected_value
        elif op == "<":
            return actual_value < expected_value
        elif op == ">=":
            return actual_value >= expected_value
        elif op == "<=":
            return actual_value <= expected_value
        elif op == "contains": # 主要用于字符串或列表
            if isinstance(actual_value, str) and isinstance(expected_value, str):
                return expected_value in actual_value
            elif isinstance(actual_value, list) and not isinstance(expected_value, list): # 检查元素是否在列表中
                 return expected_value in actual_value
            elif isinstance(actual_value, list) and isinstance(expected_value, list): # 检查是否包含所有期望的元素
                return all(item in actual_value for item in expected_value)
        elif op == "in": # expected_value 应该是一个列表
            return actual_value in expected_value
        elif op == "not in": # expected_value 应该是一个列表
            return actual_value not in expected_value
        else:
            print(f"Unsupported operator: {op} for key {key}. Condition evaluated as False.")
            return False

        return False

    def _matches_filter_group(self, run: wandb.apis.public.Run, filter_group: DictConfig) -> bool:
        """
        检查单个运行是否满足筛选组内的所有条件 (AND 关系)。
        filter_group 包含一个 'conditions' 列表。
        """
        for condition in filter_group.conditions:
            if not self._matches_condition(run, condition):
                return False # 只要有一个条件不满足，此筛选组就不通过
        return True # 所有条件都满足
    
    def read_data(self, verbose=False) -> List[wandb.apis.public.Run]:
        """
        从 WandB 获取并筛选实验运行。
        返回满足任一筛选组条件的所有运行的列表。
        """
        if self.entity:
            path = f"{self.entity}/{self.project}"
        else: # 如果 entity 未提供，wandb API 会尝试使用默认的 entity
            path = self.project
        
        print(f"Fetching all runs from WandB path: {path} ...")
        
        all_project_runs = list(self.api.runs(path=path)) # 转换为列表以避免迭代器问题
        print(f"Fetched {len(all_project_runs)} runs from project '{path}'. Now applying custom filters.")
        
        selected_runs: List[wandb.apis.public.Run] = []
        selected_run_ids = set() # 用于去重

        for run in all_project_runs:
            # 检查运行是否满足任何一个筛选组 (OR 关系)
            for filter_group in self.filters_config:
                if self._matches_filter_group(run, filter_group):
                    if run.id not in selected_run_ids:
                        selected_runs.append(run)
                        selected_run_ids.add(run.id)
                    break # 满足一个筛选组就够了，跳到下一个 run
        
        print(f"Found {len(selected_runs)} runs matching the specified filters.")
        if verbose and not selected_runs:
            print("No runs matched the filters. Double-check your filter configuration and the runs in your W&B project.")
        elif verbose and selected_runs:
            print("--- Summary of selected run IDs ---")
            for i, run in enumerate(selected_runs):
                 print(f"  {i+1:3d}. ID: {run.id}, Group: {run.group}, Name: {run.name}")
            print("----------------------------------")

        return selected_runs