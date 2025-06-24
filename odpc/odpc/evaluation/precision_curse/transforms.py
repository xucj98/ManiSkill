import numpy as np
import pandas as pd
from typing import Any, Optional, Dict, Union


class BaseTransform:
    def __init__(self, **kwargs): 
            pass
    def __call__(self, value: Any) -> Any: 
            return value


class IdentityTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, value: Any) -> Any:
        return value


class SeriesAggregationTransform(BaseTransform):
    def __init__(self, method: str, args: Optional[Dict] = None):
        super().__init__()
        self.method = method
        self.args = args or {}
    
    def __call__(self, series: pd.Series) -> Any:
        if series.empty or series.isnull().all(): return None
        series = series.dropna()
        if series.empty: return None
        if self.method == "max": return series.max()
        if self.method == "min": return series.min()
        if self.method == "mean": return series.mean()
        # ... 其他聚合 ...
        return None


class MultiplyTransform(BaseTransform):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def __call__(self, value: Union[float, int, pd.Series]) -> Any:
        return value * self.factor
    

class LogTransform(BaseTransform):
    def __init__(self, base: float = 10):
        super().__init__()
        self.base = base

    def __call__(self, value: Union[float, int, pd.Series]) -> float:
        return np.log(value) / np.log(self.base)
    

class LogOneMinusTransform(BaseTransform):
    def __init__(self, base: float = 10):
        super().__init__()
        self.base = base

    def __call__(self, value: Union[float, int, pd.Series]) -> float:
        return np.log(1 - value) / np.log(self.base)
