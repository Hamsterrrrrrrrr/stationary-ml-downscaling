# utils.py

import random
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pickle


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_historical_dataset(data_dir):
    """Load pre-processed historical dataset containing both HR and LR_interp variables."""
    print("Loading historical dataset...")
    ds_hist = xr.open_dataset(f"{data_dir}/MPI-ESM1-2-HR-LR_historical_r1i1p1f1_1850_2014_allvars.nc")
    print("  Dataset loaded successfully!")
    return ds_hist


def get_latitude_values(ds):
    """
    Extract latitude values from xarray dataset.
    
    Args:
        ds: xarray Dataset
    
    Returns:
        lat_values: numpy array of latitude values, shape (H,)
    """
    return ds['lat'].values


def load_normalization_stats(stats_path):
    """
    Load pre-computed normalization statistics.
    
    Args:
        stats_path: Path to norm_stats.pkl file
    
    Returns:
        norm_stats: Dictionary with normalization statistics
    """
    print(f"Loading normalization statistics from {stats_path}...")
    with open(stats_path, 'rb') as f:
        norm_stats = pickle.load(f)
    print("  Statistics loaded successfully!")
    return norm_stats


class InstanceNormDataset(Dataset):
    """
    Dataset with instance normalization (per-sample normalization).
    Supports both Z-score and MinMax normalization per sample.
    """
    def __init__(self, x_data, y_data, method='zscore'):
        """
        Args:
            x_data: Input data, shape (N, C, H, W)
            y_data: Target data, shape (N, C, H, W)
            method: 'zscore' or 'minmax' for normalization type
        """
        self.x_data = x_data
        self.y_data = y_data
        self.method = method
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]  # Shape: (C, H, W)
        y = self.y_data[idx]  # Shape: (C, H, W)
        
        if self.method == 'zscore':
            # Z-score normalization per sample
            # Compute stats from input for each channel
            mean = x.mean(dim=(1, 2), keepdim=True)  # Shape: (C, 1, 1)
            std = x.std(dim=(1, 2), keepdim=True)     # Shape: (C, 1, 1)
            
            # Normalize both input and target using input's stats
            x_norm = (x - mean) / (std + 1e-8)
            y_norm = (y - mean) / (std + 1e-8)
            
        elif self.method == 'minmax':
            # MinMax normalization per sample to [-1, 1]
            # Compute min/max from input for each channel
            x_min = x.amin(dim=(1, 2), keepdim=True)  # Shape: (C, 1, 1)
            x_max = x.amax(dim=(1, 2), keepdim=True)  # Shape: (C, 1, 1)
            
            # Normalize to [-1, 1]
            x_range = x_max - x_min
            x_norm = 2 * (x - x_min) / (x_range + 1e-8) - 1
            y_norm = 2 * (y - x_min) / (x_range + 1e-8) - 1
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        return x_norm, y_norm


def create_dataloaders(ds, variables, norm_stats, 
                       train_period=('1850', '1980'),
                       val_period=('1981', '2000'),
                       batch_size=128,
                       normalization='minmax_global'):
    """
    Create train/val dataloaders with normalization.
    
    Args:
        ds: xarray Dataset containing both HR and LR_interp variables
        variables: List of HR variable names (e.g., ['pr_hr', 'tas_hr'])
        norm_stats: Normalization statistics (loaded from pickle)
        train_period, val_period: Time periods for train/val splits
        batch_size: Batch size for DataLoader
        normalization: One of:
            - 'minmax_global': Min-max using global scalars to [-1, 1]
            - 'minmax_pixel': Min-max using per-pixel 2D stats to [-1, 1]
            - 'zscore_global': Z-score using global mean/std
            - 'zscore_pixel': Z-score using per-pixel mean/std
            - 'instance_zscore': Per-sample Z-score normalization
            - 'instance_minmax': Per-sample MinMax normalization to [-1, 1]
            - 'none': No normalization (raw data)
    
    Returns:
        train_loader, val_loader
    """

    x_train_list = []
    x_val_list = []
    y_train_list = []
    y_val_list = []
    
    for var in variables:
        print(f"Processing {var}...")
        
        # Get base variable name and LR variable name
        var_base = var.replace('_hr', '')
        var_lr = var.replace('_hr', '_lr_interp')
        
        # Extract train/val data - CONVERT TO NUMPY IMMEDIATELY
        hr_train = ds[var].sel(time=slice(train_period[0], train_period[1])).values
        hr_val = ds[var].sel(time=slice(val_period[0], val_period[1])).values
        lr_train = ds[var_lr].sel(time=slice(train_period[0], train_period[1])).values
        lr_val = ds[var_lr].sel(time=slice(val_period[0], val_period[1])).values
        
        # Normalize based on method
        if normalization == 'none':
            # No normalization - return raw data
            x_train = lr_train
            x_val = lr_val
            y_train = hr_train
            y_val = hr_val
        
        elif normalization in ['instance_zscore', 'instance_minmax']:
            # For instance norm, use raw data (normalization happens in Dataset)
            x_train = lr_train
            x_val = lr_val
            y_train = hr_train
            y_val = hr_val
            
        elif normalization == 'minmax_global':
            # Min-max normalization using global scalars to [-1, 1]
            hr_min = norm_stats[var_base]['hr']['global_min']
            hr_max = norm_stats[var_base]['hr']['global_max']
            lr_min = norm_stats[var_base]['lr_interp']['global_min']
            lr_max = norm_stats[var_base]['lr_interp']['global_max']
            
            # Normalize to [-1, 1]
            x_train = 2 * (lr_train - lr_min) / (lr_max - lr_min + 1e-8) - 1
            x_val = 2 * (lr_val - lr_min) / (lr_max - lr_min + 1e-8) - 1
            y_train = 2 * (hr_train - hr_min) / (hr_max - hr_min + 1e-8) - 1
            y_val = 2 * (hr_val - hr_min) / (hr_max - hr_min + 1e-8) - 1
            
        elif normalization == 'minmax_pixel':
            # Min-max normalization using per-pixel 2D stats to [-1, 1]
            hr_min = norm_stats[var_base]['hr']['pixel_min']
            hr_max = norm_stats[var_base]['hr']['pixel_max'] 
            lr_min = norm_stats[var_base]['lr_interp']['pixel_min']  
            lr_max = norm_stats[var_base]['lr_interp']['pixel_max'] 
            
            # Normalize to [-1, 1]
            x_train = 2 * (lr_train - lr_min) / (lr_max - lr_min + 1e-8) - 1
            x_val = 2 * (lr_val - lr_min) / (lr_max - lr_min + 1e-8) - 1
            y_train = 2 * (hr_train - hr_min) / (hr_max - hr_min + 1e-8) - 1
            y_val = 2 * (hr_val - hr_min) / (hr_max - hr_min + 1e-8) - 1
            
        elif normalization == 'zscore_global':
            # Z-score normalization using global mean/std
            hr_mean = norm_stats[var_base]['hr']['global_mean']
            hr_std = norm_stats[var_base]['hr']['global_std']
            lr_mean = norm_stats[var_base]['lr_interp']['global_mean']
            lr_std = norm_stats[var_base]['lr_interp']['global_std']
            
            x_train = (lr_train - lr_mean) / (lr_std + 1e-8)
            x_val = (lr_val - lr_mean) / (lr_std + 1e-8)
            y_train = (hr_train - hr_mean) / (hr_std + 1e-8)
            y_val = (hr_val - hr_mean) / (hr_std + 1e-8)
            
        elif normalization == 'zscore_pixel':
            # Z-score normalization using per-pixel mean/std
            hr_mean = norm_stats[var_base]['hr']['pixel_mean']
            hr_std = norm_stats[var_base]['hr']['pixel_std']    
            lr_mean = norm_stats[var_base]['lr_interp']['pixel_mean']  
            lr_std = norm_stats[var_base]['lr_interp']['pixel_std']    
            
            x_train = (lr_train - lr_mean) / (lr_std + 1e-8)
            x_val = (lr_val - lr_mean) / (lr_std + 1e-8)
            y_train = (hr_train - hr_mean) / (hr_std + 1e-8)
            y_val = (hr_val - hr_mean) / (hr_std + 1e-8)
        
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")
        
        x_train_list.append(x_train)
        x_val_list.append(x_val)
        y_train_list.append(y_train)
        y_val_list.append(y_val)
    
    # Stack channels if multiple variables
    if len(variables) == 1:
        x_train = np.expand_dims(x_train_list[0], axis=1)
        x_val = np.expand_dims(x_val_list[0], axis=1)
        y_train = np.expand_dims(y_train_list[0], axis=1)
        y_val = np.expand_dims(y_val_list[0], axis=1)
    else:
        x_train = np.stack(x_train_list, axis=1)
        x_val = np.stack(x_val_list, axis=1)
        y_train = np.stack(y_train_list, axis=1)
        y_val = np.stack(y_val_list, axis=1)
    
    print(f"\nDataset shapes:")
    print(f"  x_train: {x_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  x_val: {x_val.shape}")
    print(f"  y_val: {y_val.shape}")
    
    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    # Create datasets
    if normalization == 'instance_zscore':
        train_dataset = InstanceNormDataset(x_train, y_train, method='zscore')
        val_dataset = InstanceNormDataset(x_val, y_val, method='zscore')
    elif normalization == 'instance_minmax':
        train_dataset = InstanceNormDataset(x_train, y_train, method='minmax')
        val_dataset = InstanceNormDataset(x_val, y_val, method='minmax')
    else:
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader