import wandb
import pandas as pd
from typing import List, Dict, Any, Optional
from omegaconf import DictConfig

from odpc.utils.utils import instantiate_from_config, get_nested_value
from odpc.evaluation.precision_curse.transforms import BaseTransform

class WandbProcessor:
    def __init__(
            self,
            data_schema: DictConfig,
    ):
        self.data_schema = data_schema
        
        # 预先实例化所有的转换器（如果配置中定义了）
        self.transforms_cache: Dict[str, Optional[BaseTransform]] = {}
        
        for field_key, field_cfg in self.data_schema.items():
            transform_instance = None
            if field_cfg and "transform" in field_cfg and field_cfg.transform:
                transform_instance: BaseTransform = instantiate_from_config(field_cfg.transform)
            self.transforms_cache[field_key] = transform_instance

        print(f"WandbProcessor initialized. Schema fields: {list(data_schema.keys())}")
    
    def _extract_field_value(self, run: wandb.apis.public.Run, field_name_in_schema: str, field_config: DictConfig) -> Any:
        """
        从单个 run 对象中提取并转换单个字段的值。
        run: wandb.apis.public.Run 对象
        field_name_in_schema: schema 中定义的字段名 (如 "data_quantity")
        field_config: 该字段的配置 (包含 source_type, path_or_key, transform 等)
        """
        source_type = field_config.source_type
        path_or_key = field_config.path_or_key
        raw_value = None

        # 1. 提取原始值
        if source_type == "config":
            raw_value = get_nested_value(run.config, path_or_key)
        elif source_type == "summary":
            raw_value = get_nested_value(run.summary, path_or_key)
        elif source_type == "history":
            history_df = run.history(keys=[path_or_key], pandas=True, stream="default") # stream="default" 用于主要指标流
            raw_value = history_df[path_or_key].dropna() # 获取Series并移除NaN
        elif source_type == "attribute": # 直接访问 run 的属性
            raw_value = getattr(run, path_or_key)
        else:
            raise ValueError(f"Unknown source_type '{source_type}' for field '{field_name_in_schema}' in run '{run.id}'.")
        
        # 2. 应用转换 (如果定义了)
        final_value = raw_value
        transform_instance = self.transforms_cache.get(field_name_in_schema)

        if transform_instance:
            final_value = transform_instance(raw_value)

        return final_value


    def process_data(self, raw_runs_data: List[wandb.apis.public.Run], verbose: bool = False) -> pd.DataFrame:
        """
        将从WandbReader获取的原始运行数据列表转换为结构化的Pandas DataFrame。
        raw_runs_data: List[wandb.apis.public.Run]
        """
        print(f"Processing {len(raw_runs_data)} runs...")
        processed_records = []

        for run in raw_runs_data:
            record = {}
            valid_run_data = True # 标记此运行的数据是否有效（所有必需字段都有值）

            for field_key, field_cfg in self.data_schema.items():
                try:
                    extracted_value = self._extract_field_value(run, field_key, field_cfg)
                except Exception as e:
                    if verbose:
                        print(f"  ERROR: Failed to extract value for field '{field_key}' in run '{run.id}'. Error: {e}")
                    extracted_value = None
                
                record[field_key] = extracted_value

                # 处理缺失值
                if extracted_value is None:
                    on_missing_strategy = field_cfg.get("on_missing", "drop_run")
                    if on_missing_strategy == "drop_run":
                        if verbose:
                            print(f"Run '{run.id}' will be dropped because required field '{field_key}' (from schema key '{field_key}') is missing.")
                        valid_run_data = False
                        break # 此运行无效，不再处理其他字段
                    elif on_missing_strategy == "fill_value":
                        fill_val = field_cfg.get("fill_value")
                        record[field_key] = fill_val
                        if verbose:
                            print(f"Missing value for '{field_key}' in run '{run.id}', filled with: {fill_val}")
                   
            if valid_run_data:
                processed_records.append(record)
                if verbose:
                    print(f"    SUCCESS: Processed run '{run.id}'. Data: { record }")

        if not processed_records:
            print("No records were successfully processed. Returning empty DataFrame.")
            return pd.DataFrame()

        df = pd.DataFrame(processed_records)
        
        print(f"Processor finished. Output DataFrame shape: {df.shape}")
        if verbose:
            print(df)
        
        return df
