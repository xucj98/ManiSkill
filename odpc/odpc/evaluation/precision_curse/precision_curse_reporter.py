import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, List, Optional, Callable

from odpc.utils.utils import instantiate_from_config
from odpc.evaluation.precision_curse.transforms import BaseTransform, IdentityTransform


class PrecisionCurseReporter:
    def __init__(
            self,
            display_names_map: Optional[DictConfig] = None,
            target_metric_values_for_estimation: Optional[List[float]] = None,
            plot_metric_vs_num_data: Optional[DictConfig] = None,
            # plot_estimated_data_vs_precision: Optional[DictConfig] = None, # 暂时不实现第二组图
            output_options: Optional[DictConfig] = None,
    ):

        self.display_names = display_names_map
        self.target_metrics = target_metric_values_for_estimation

        self.cfg_plot_metric_vs_num_data = plot_metric_vs_num_data # nd for num_data
        self.cfg_output = output_options

        self.save_dir = output_options.save_dir
        os.makedirs(self.save_dir, exist_ok=True)        

        # 预实例化转换器
        self.x_transformer: Optional[BaseTransform] = None
        self.y_transformer: Optional[BaseTransform] = None
        if self.cfg_plot_metric_vs_num_data.x_axis.get("transform"):
            self.x_transformer = instantiate_from_config(self.cfg_plot_metric_vs_num_data.x_axis.transform)
        else:
            self.x_transformer = IdentityTransform()
        if self.cfg_plot_metric_vs_num_data.y_axis.get("transform"):
            self.y_transformer = instantiate_from_config(self.cfg_plot_metric_vs_num_data.y_axis.transform)
        else:
            self.y_transformer = IdentityTransform()

        print("PrecisionCurseReporter initialized.")

    def _get_display_name(self, key: str, default: Optional[str] = None) -> str:
        """获取显示名，如果映射中没有，则返回key本身或提供的默认值"""
        name = self.display_names.get(key, default if default else key)
        return str(name)


    def _plot_metric_vs_num_data_single_group(
        self, 
        ax,
        group_df: pd.DataFrame,
        group_name: Any,
        plot_config: DictConfig
    ):
        """为单个精度组绘制 Metric vs NumData"""
        original_x_data = group_df["num_data"]
        original_y_data = group_df["metric"]

        transformed_x_data = self.x_transformer(original_x_data)
        transformed_y_data = self.y_transformer(original_y_data)

        # 过滤掉转换后可能产生的 NaN/inf 值，这些值无法绘图或拟合
        valid_indices = np.isfinite(transformed_x_data) & np.isfinite(transformed_y_data)
        if not valid_indices.any():
            print(f"  WARNING: No valid (finite) data points after transformation for group '{group_name}'. Skipping.")
            return
        
        plot_x = transformed_x_data[valid_indices]
        plot_y = transformed_y_data[valid_indices]
        # 对于后续可能需要的原始值（例如，如果回归后要在原始空间显示方程）
        # original_x_for_fit = original_x_data[valid_indices]
        # original_y_for_fit = original_y_data[valid_indices]

        # 图例标签 (只为第一个绘制元素添加，后续用 _nolegend_)
        # 使用 display_name 获取 "precision" 的显示名
        precision_display_name = self._get_display_name("precision", "Precision")
        legend_label = f"{precision_display_name} = {group_name}"

        # 2. 绘制元素
        for element_cfg in plot_config.get("plot_elements", []): # 确保 plot_elements存在
            el_type = element_cfg.type
            if el_type == "scatter":
                ax.scatter(plot_x, plot_y, 
                           label=legend_label,
                           marker=element_cfg.get("marker", "o"),
                           s=element_cfg.get("size", 30),
                           alpha=element_cfg.get("alpha", 0.7))
                legend_label = "_nolegend_" # 后续元素不再重复图例标签

            elif el_type == "linear_regression_fit":
                if len(plot_x) < 2: # 至少需要两个点才能做线性回归
                    print(f"  INFO: Not enough data points ({len(plot_x)}) for linear regression in group '{group_name}'.")
                    continue

                slope, intercept, r_value, p_value, std_err = stats.linregress(plot_x, plot_y)

                # 绘制回归线
                x_fit_line = np.array([plot_x.min(), plot_x.max()])
                y_fit_line = slope * x_fit_line + intercept
                ax.plot(x_fit_line, y_fit_line, linestyle="--", label=legend_label)
                legend_label = "_nolegend_"

                # 显示方程和 R²
                eq_parts = []
                if element_cfg.get("display_equation", False):
                    # 注意：这里的方程是针对转换后空间的 Y 和 X
                    # 如果X是log(num_data)，Y是log(1-metric)，方程是 log(1-metric) = slope * log(num_data) + intercept
                    # 转换回原始方程会很复杂，通常显示转换后空间的线性方程
                    var_y_transformed = plot_config.y_axis.transform_label
                    var_x_transformed = plot_config.x_axis.transform_label
                    eq_parts.append(f"{var_y_transformed} ≈ {slope:.2f} * {var_x_transformed} + {intercept:.2f}")
                if element_cfg.get("display_r_squared", False):
                    eq_parts.append(f"R² = {r_value**2:.2f}")
                
                if eq_parts:
                    # 在图上找个合适的位置放置文本，或添加到图例
                    # 简单的做法：在回归线上方某处
                    text_x = x_fit_line.mean()
                    text_y = slope * text_x + intercept + 0.05 * (plot_y.max() - plot_y.min()) # 稍微偏移
                    ax.text(text_x, text_y, "\n".join(eq_parts), fontsize=8, ha='center')


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


    def plot_metric_vs_num_data(self, df: pd.DataFrame):
        """绘制第一组图：Metric vs NumData, 按 Precision 分组"""
        plot_cfg = self.cfg_plot_metric_vs_num_data
        grouping_key = "precision"

        fig, ax = plt.subplots(figsize=plot_cfg.get("figsize", (10, 6)))

        grouped = df.groupby(grouping_key)
        for name, group_data in grouped:
            self._plot_metric_vs_num_data_single_group(ax, group_data.copy(), name, plot_cfg)

        ax.set_title(plot_cfg.title)

        # 设置X轴和Y轴的标签和刻度
        self._setup_axis_ticks_and_labels(ax, plot_cfg.x_axis, self.x_transformer, is_x_axis=True)
        self._setup_axis_ticks_and_labels(ax, plot_cfg.y_axis, self.y_transformer, is_x_axis=False)

        ax.legend(title=plot_cfg.grouping_legend_title)

        ax.grid(True, which="both", ls="-", alpha=0.5)
        fig.tight_layout()

        if self.cfg_output and self.cfg_output.get("save_plots", False):
            filename = f"metric_vs_num_data.{self.cfg_output.get('plot_format', 'png')}"
            filepath = os.path.join(self.save_dir, filename)
            fig.savefig(filepath)
            print(f"Plot saved to {filepath}")
        
        plt.show() # 根据需要决定是否总是显示

    def report(self, processed_df: pd.DataFrame, verbose: bool = False): # 添加verbose
        print("Reporter started...")
     
        # 保存处理后的数据 (如果配置了)
        if self.cfg_output and self.cfg_output.get("save_processed_data_csv", False):
            csv_path = os.path.join(self.save_dir, "processed_data.csv")
            processed_df.to_csv(csv_path, index=False)
            print(f"Processed data saved to {csv_path}")

        # 调用绘图函数
        self.plot_metric_vs_num_data(processed_df)
        
        print("Reporter finished.")