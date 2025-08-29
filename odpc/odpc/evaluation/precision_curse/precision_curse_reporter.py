import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, List, Optional, Callable

from odpc.utils.utils import instantiate_from_config
from odpc.evaluation.precision_curse.transforms import BaseTransform, IdentityTransform, LogTransform, InverseShiftedTransform


class PrecisionCurseReporter:
    def __init__(
            self,
            display_names_map: Optional[DictConfig] = None,
            target_metrics: Optional[List[float]] = None,
            report_metric_vs_num_data: Optional[DictConfig] = None,
            report_num_data_vs_precision: Optional[DictConfig] = None,
            output_options: Optional[DictConfig] = None,
    ):

        self.display_names = display_names_map
        self.target_metrics = target_metrics

        self.cfg_report_metric_vs_num_data = report_metric_vs_num_data
        self.cfg_report_num_data_vs_precision = report_num_data_vs_precision
        self.cfg_output = output_options

        self.save_dir = output_options.save_dir
        os.makedirs(self.save_dir, exist_ok=True)        

        # 预实例化转换器
        self.x_transformer: BaseTransform = IdentityTransform()
        self.y_transformer: BaseTransform = IdentityTransform()
        if self.cfg_report_metric_vs_num_data.x_axis.get("transform"):
            self.x_transformer = instantiate_from_config(self.cfg_report_metric_vs_num_data.x_axis.transform)
        if self.cfg_report_metric_vs_num_data.y_axis.get("transform"):
            self.y_transformer = instantiate_from_config(self.cfg_report_metric_vs_num_data.y_axis.transform)

        print("PrecisionCurseReporter initialized.")

    def _get_display_name(self, key: str, default: Optional[str] = None) -> str:
        """获取显示名，如果映射中没有，则返回key本身或提供的默认值"""
        name = self.display_names.get(key, default if default else key)
        return str(name)

    def _setup_axis_ticks_and_labels(
        self,
        ax,
        axis_cfg: DictConfig, # x_axis 或 y_axis 的配置
        transformer: Optional[BaseTransform],
        is_x_axis: bool
    ):
        """
        设置坐标轴的标签和刻度。
        假设 axis_cfg.ticks 总是由用户在配置文件中提供。
        """
        # 1. 设置轴标签
        ax.set_xlabel(axis_cfg.label) if is_x_axis else ax.set_ylabel(axis_cfg.label)

        # 2. 获取用户在配置中指定的原始刻度点
        original_ticks_from_config = OmegaConf.to_container(axis_cfg.ticks) # 转为 Python list
        original_ticks_numeric = np.array(original_ticks_from_config)

        # 3. 使用转换器将原始刻度点转换为绘图空间的位置
        tick_positions_transformed = transformer(original_ticks_numeric)

        # 4. 准备刻度标签 (原始刻度值的字符串形式)
        tick_labels_str = [f"{val:.3g}" for val in original_ticks_numeric] # .3g 保留3位有效数字

        # 5. 应用刻度和标签
        if is_x_axis:
            ax.set_xticks(tick_positions_transformed)
            ax.set_xticklabels(tick_labels_str)
        else:
            ax.set_yticks(tick_positions_transformed)
            ax.set_yticklabels(tick_labels_str)

    def _report_single_group(
        self,
        ax: plt.Axes,
        original_x: pd.Series,
        original_y: pd.Series,
        x_transformer: BaseTransform,
        y_transformer: BaseTransform,
        plot_config: DictConfig,
        legend_label: str,
    ) -> Optional[Dict[str, float]]:
        """为单个组绘制图表"""
        transformed_x = x_transformer(original_x)
        transformed_y = y_transformer(original_y)

        # 过滤掉转换后可能产生的 NaN/inf 值
        valid_indices = np.isfinite(transformed_x) & np.isfinite(transformed_y)
        if not valid_indices.any():
            return  

        plot_x = transformed_x[valid_indices]
        plot_y = transformed_y[valid_indices]

        # 2. 计算线性回归参数 (在转换后的空间)
        regression_results: Optional[Dict[str, float]] = None
        if len(plot_x) >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(plot_x, plot_y)  
            regression_results = {
                "slope": slope,
                "intercept": intercept,
                "r_value": r_value,
                "p_value": p_value,
                "std_err": std_err
            }

        # 3. 绘制元素 (根据 plot_elements 配置)
        for element_cfg in plot_config.get("plot_elements", []):
            el_type = element_cfg.type
            
            if el_type == "scatter":
                ax.scatter(
                    plot_x, plot_y, 
                    label=legend_label, 
                    marker=element_cfg.get("marker", "o"), 
                    s=element_cfg.get("size", 30), 
                    alpha=element_cfg.get("alpha", 0.7))
                
            elif el_type == "linear_regression_fit":
                if regression_results is None:
                    print(f"  INFO: Not enough data points ({len(plot_x)}) for linear regression in group '{legend_label}'.")
                    continue
                else:
                    slope = regression_results["slope"]
                    intercept = regression_results["intercept"]
                    r_value = regression_results["r_value"]

                # 绘制回归线
                x_fit_line = np.array([plot_x.min(), plot_x.max()])
                y_fit_line = slope * x_fit_line + intercept
                ax.plot(x_fit_line, y_fit_line, linestyle="--", label=legend_label)
                
                # 显示方程和 R²
                eq_parts = []
                if element_cfg.get("display_equation", False):
                    var_y = plot_config.y_axis.transformed_label
                    var_x = plot_config.x_axis.transformed_label
                    eq_parts.append(f"{var_y} ≈ {slope:.2f} * {var_x} + {intercept:.2f}")
                if element_cfg.get("display_r_squared", False):
                    eq_parts.append(f"R² = {r_value**2:.2f}")

                if eq_parts:
                    # 在图上找个合适的位置放置文本，或添加到图例
                    # 简单的做法：在回归线上方某处
                    text_x = x_fit_line.mean()
                    text_y = slope * text_x + intercept + 0.05 * (plot_y.max() - plot_y.min()) # 稍微偏移
                    ax.text(text_x, text_y, "\n".join(eq_parts), fontsize=8, ha='center')
                
            legend_label = "_nolegend_" # 后续元素不再重复图例标签

        return regression_results
    
    def _report_groups(
            self, 
            df: pd.DataFrame, 
            group_key: str, x_key: str, y_key: str,
            x_transformer: BaseTransform, y_transformer: BaseTransform,
            plot_cfg: DictConfig,
    ):
        fig, ax = plt.subplots(figsize=plot_cfg.get("figsize", (10, 6)))
        grouped = df.groupby(group_key)
        results = []
        for name, group_data in grouped:
            regression_results = self._report_single_group(
                ax, 
                group_data[x_key], group_data[y_key], 
                x_transformer, y_transformer, 
                plot_cfg, 
                f"{self.display_names.get(group_key, group_key)} = {name}"
            )
            if regression_results:
                regression_results[group_key] = name
                results.append(regression_results)

        ax.set_title(plot_cfg.title)
        self._setup_axis_ticks_and_labels(ax, plot_cfg.x_axis, x_transformer, is_x_axis=True)
        self._setup_axis_ticks_and_labels(ax, plot_cfg.y_axis, y_transformer, is_x_axis=False)

        ax.legend(title=plot_cfg.grouping_legend_title)

        ax.grid(True, which="both", ls="-", alpha=0.5)
        fig.tight_layout()

        if self.cfg_output and self.cfg_output.get("save_plots", False):
            filename = f"{plot_cfg.title.replace(' ', '_')}.{self.cfg_output.get('plot_format', 'png')}"
            filepath = os.path.join(self.save_dir, filename)
            fig.savefig(filepath)
            print(f"Plot saved to {filepath}")

        return pd.DataFrame(results)

    def report_metric_vs_num_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """绘制第一组图：Metric vs NumData, 按 Precision 分组"""
        return self._report_groups(
            df, 
            "precision", "num_data", "metric", 
            self.x_transformer, self.y_transformer, 
            self.cfg_report_metric_vs_num_data,
        )
    
    def _estimate_num_data_for_target_metrics(self, regression_params: pd.DataFrame) -> pd.DataFrame:
        """
        根据回归参数和目标性能指标，估计所需的数据量。
        """
        
        results = []
        for target_metric in self.target_metrics:
            transformed_target_metric = self.y_transformer(target_metric)
            
            for params in regression_params.itertuples():
                precision = params.precision
                slope = params.slope
                intercept = params.intercept

                try:
                    transformed_estimated_nd = (transformed_target_metric - intercept) / slope
                    estimated_nd_orig = self.x_transformer.inverse(transformed_estimated_nd)
                except Exception as e:
                    estimated_nd_orig = np.nan

                if pd.isna(estimated_nd_orig):
                    print(f"  INFO: For precision {precision}, target {target_metric}: estimated num_data is NaN. Drop data for this precision.")
                    continue
                
                results.append({
                    "metric": target_metric,
                    "precision": precision,
                    "estimated_num_data": estimated_nd_orig,
                })
        
        return pd.DataFrame(results)
    
    # 新的绘图函数，用于绘制第二组图
    def report_estimated_data_vs_precision(self, df: pd.DataFrame):
        """
        绘制第二组图：估计的 NumData vs. Precision, 按 TargetMetric 分组。
        df_estimated 包含列: "metric", "precision", "estimated_num_data"
        """
        plot_cfg = self.cfg_report_num_data_vs_precision
        
        plot2_x_transformer: BaseTransform = IdentityTransform()
        plot2_y_transformer: BaseTransform = IdentityTransform()
        if plot_cfg.x_axis.get("transform"):
            plot2_x_transformer = instantiate_from_config(plot_cfg.x_axis.transform)
        if plot_cfg.y_axis.get("transform"):
            plot2_y_transformer = instantiate_from_config(plot_cfg.y_axis.transform)

        return self._report_groups(
            df, 
            "metric", "precision", "estimated_num_data", 
            plot2_x_transformer, plot2_y_transformer, 
            plot_cfg,
        )
    
    def search_precision_limit(self, df: pd.DataFrame, search_range: list) -> float:
        grouped = df.groupby("metric")
        best_r, best_c = 0., 0.
        for shift in np.arange(*search_range):
            x_transformer = InverseShiftedTransform(shift=shift)
            y_transformer = LogTransform(base=2)
            sum_r = 0
            for name, group_data in grouped:
                x_data = group_data["precision"]
                y_data = group_data["estimated_num_data"]
                transformed_x = x_transformer(x_data)
                transformed_y = y_transformer(y_data)
                slope, intercept, r_value, p_value, std_err = stats.linregress(transformed_x, transformed_y)
                sum_r += r_value
            if sum_r > best_r:
                best_r = sum_r
                best_c = shift
        return float(best_c)

    def report(self, processed_df: pd.DataFrame, verbose: bool = False): # 添加verbose
        print("Reporter started...")
     
        if self.cfg_output and self.cfg_output.get("save_processed_data_csv", False):
            csv_path = os.path.join(self.save_dir, "raw_data.csv")
            processed_df.to_csv(csv_path, index=False)
            print(f"Raw data saved to {csv_path}")

        metric_vs_num_data_results = self.report_metric_vs_num_data(processed_df)
        if verbose:
            print("=========== Regression results of Metric vs NumData ===========")
            print(metric_vs_num_data_results)
            print("===============================================================")
        if self.cfg_output and self.cfg_output.get("save_processed_data_csv", False):
            csv_path = os.path.join(self.save_dir, "metric_vs_num_data_results.csv")
            metric_vs_num_data_results.to_csv(csv_path, index=False)
            print(f"Metric vs NumData results saved to {csv_path}")

        estimated_num_data_results = self._estimate_num_data_for_target_metrics(metric_vs_num_data_results)
        if verbose:
            print("=========== Estimated numData for target metrics ===========")
            print(estimated_num_data_results)
            print("============================================================")
        if self.cfg_output and self.cfg_output.get("save_processed_data_csv", False):
            csv_path = os.path.join(self.save_dir, "estimated_num_data_results.csv")
            estimated_num_data_results.to_csv(csv_path, index=False)
            print(f"Estimated numData results saved to {csv_path}")

        if hasattr(self.cfg_report_num_data_vs_precision, "search_range"):
            search_range = self.cfg_report_num_data_vs_precision.search_range
            precision_limit = self.search_precision_limit(estimated_num_data_results, search_range)
            print(f"precision_limit: {precision_limit:.2f}")
            self.cfg_report_num_data_vs_precision.precision_limit = precision_limit

        self.report_estimated_data_vs_precision(estimated_num_data_results)

        print("Reporter finished.")

        plt.show()