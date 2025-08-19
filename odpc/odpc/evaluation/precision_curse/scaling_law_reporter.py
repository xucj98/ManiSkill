import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, List, Optional, Callable

from odpc.utils.utils import instantiate_from_config
from odpc.evaluation.precision_curse.transforms import BaseTransform, IdentityTransform
from odpc.evaluation.precision_curse.precision_curse_reporter import PrecisionCurseReporter

class ScalingLawReporter(PrecisionCurseReporter):
    def __init__(
            self, 
            display_names_map: Optional[DictConfig] = None,
            report_metric_vs_num_data: Optional[DictConfig] = None,
            output_options: Optional[DictConfig] = None,
    ):
        super().__init__(
            display_names_map=display_names_map,
            report_metric_vs_num_data=report_metric_vs_num_data,
            output_options=output_options,
        )

    def report(self, processed_df: pd.DataFrame, verbose: bool = False):
        print("Reporter started...")
     
        if self.cfg_output and self.cfg_output.get("save_processed_data_csv", False):
            csv_path = os.path.join(self.save_dir, "raw_data.csv")
            processed_df.to_csv(csv_path, index=False)
            print(f"Raw data saved to {csv_path}")

        metric_vs_num_data_results = self._report_groups(
            processed_df,
            "group_name",
            "num_data",
            "metric",
            self.x_transformer,
            self.y_transformer,
            self.cfg_report_metric_vs_num_data,
        )

        print("Reporter finished.")

        plt.show()
        