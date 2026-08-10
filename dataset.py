import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple

class TabularDataset(Dataset):
    def __init__(self, X: Union[np.ndarray, pd.DataFrame, torch.Tensor], 
                 y: Optional[Union[np.ndarray, pd.Series, torch.Tensor]] = None, 
                 y_dtype: torch.dtype = torch.long):
        self.X = self._convert_to_tensor(X, torch.float32, "X")
        self.y = self._convert_to_tensor(y, y_dtype, "y") if y is not None else None
        
        if self.X.ndim != 2:
            raise ValueError(f"X must be a 2D array/tensor, got shape {self.X.shape}")
        if self.y is not None and len(self.X) != len(self.y):
            raise ValueError(f"X and y must have the same number of samples. X: {len(self.X)}, y: {len(self.y)}")
        
        if torch.isnan(self.X).any() or torch.isinf(self.X).any():
            raise ValueError("X contains NaN or Inf values. Please preprocess data.")
        if self.y is not None and (torch.isnan(self.y.float()).any() or torch.isinf(self.y.float()).any()):
            raise ValueError("y contains NaN or Inf values. Please preprocess data.")

    def _convert_to_tensor(self, data: Union[np.ndarray, pd.DataFrame, torch.Tensor], 
                          dtype: torch.dtype, name: str) -> torch.Tensor:
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data = data.values
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(dtype)
        elif isinstance(data, torch.Tensor):
            data = data.to(dtype)
        else:
            raise TypeError(f"{name} must be a numpy array, pandas DataFrame/Series, or torch Tensor, got {type(data)}")
        
        if data.ndim == 1:
            data = data.view(-1, 1)
        
        return data

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.X[idx]
        if self.y is not None:
            y = self.y[idx]
            return x, y
        return x