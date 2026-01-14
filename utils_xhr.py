# utils_xhr.py

import random
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader, TensorDataset
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


def load_xhr_data(data_dir='data'):
    """
    Load preprocessed XHR data from separate .nc files.
    
    Returns:
        dict with keys: 'train_input', 'val_input', 'test_input',
                        'train_target', 'val_target', 'train_era5', 'val_era5'
    """
    print("Loading XHR preprocessed data...")
    
    data = {
        # Detrended inputs
        'train_input': xr.open_dataarray(f"{data_dir}/cmip6/hist_train_detrend_xhr.nc"),
        'val_input': xr.open_dataarray(f"{data_dir}/cmip6/hist_val_detrend_xhr.nc"),
        'test_input': xr.open_dataarray(f"{data_dir}/cmip6/g6_test_detrend_xhr.nc"),
        
        # Residual targets (not detrended)
        'train_target': xr.open_dataarray(f"{data_dir}/era5/residual_train_xhr.nc"),
        'val_target': xr.open_dataarray(f"{data_dir}/era5/residual_val_xhr.nc"),
        
        # ERA5 reference
        'train_era5': xr.open_dataarray(f"{data_dir}/era5/era5_train_xhr.nc"),
        'val_era5': xr.open_dataarray(f"{data_dir}/era5/era5_val_xhr.nc"),
    }
    
    print("  Data loaded successfully!")
    for key, val in data.items():
        print(f"    {key}: {val.shape}")
    
    return data


def load_normalization_stats_xhr(stats_path='data/norm_stats_xhr.pkl'):
    """Load pre-computed XHR normalization statistics."""
    print(f"Loading normalization statistics from {stats_path}...")
    with open(stats_path, 'rb') as f:
        norm_stats = pickle.load(f)
    print("  Statistics loaded successfully!")
    return norm_stats


def get_latitude_values(data):
    """Extract latitude values from xarray DataArray."""
    return data['lat'].values


def normalize_zscore_pixel(data, mean, std):
    """
    Apply pixel-wise z-score normalization.
    
    Args:
        data: numpy array (T, H, W)
        mean: numpy array (H, W)
        std: numpy array (H, W)
    
    Returns:
        normalized data (T, H, W)
    """
    return (data - mean) / (std + 1e-8)


def denormalize_zscore_pixel(data, mean, std):
    """
    Reverse pixel-wise z-score normalization.
    
    Args:
        data: numpy array (T, H, W) or torch tensor
        mean: numpy array (H, W)
        std: numpy array (H, W)
    
    Returns:
        denormalized data
    """
    return data * (std + 1e-8) + mean


def create_xhr_dataloaders(data, norm_stats, batch_size=32):
    """
    Create train/val dataloaders for XHR residual prediction.
    
    Args:
        data: dict from load_xhr_data()
        norm_stats: dict from load_normalization_stats_xhr()
        batch_size: batch size for DataLoader
    
    Returns:
        train_loader, val_loader
    """
    print("\nCreating XHR dataloaders...")
    
    # Extract numpy arrays
    x_train = data['train_input'].values  # (T, H, W)
    x_val = data['val_input'].values
    y_train = data['train_target'].values
    y_val = data['val_target'].values
    
    # Apply z-score pixel normalization
    print("  Applying z-score pixel normalization...")
    
    x_train = normalize_zscore_pixel(
        x_train, 
        norm_stats['input_detrend']['mean'], 
        norm_stats['input_detrend']['std']
    )
    x_val = normalize_zscore_pixel(
        x_val, 
        norm_stats['input_detrend']['mean'], 
        norm_stats['input_detrend']['std']
    )
    y_train = normalize_zscore_pixel(
        y_train, 
        norm_stats['residual']['mean'], 
        norm_stats['residual']['std']
    )
    y_val = normalize_zscore_pixel(
        y_val, 
        norm_stats['residual']['mean'], 
        norm_stats['residual']['std']
    )
    
    # Add channel dimension: (T, H, W) -> (T, 1, H, W)
    x_train = np.expand_dims(x_train, axis=1)
    x_val = np.expand_dims(x_val, axis=1)
    y_train = np.expand_dims(y_train, axis=1)
    y_val = np.expand_dims(y_val, axis=1)
    
    print(f"\n  Dataset shapes:")
    print(f"    x_train: {x_train.shape}")
    print(f"    y_train: {y_train.shape}")
    print(f"    x_val: {x_val.shape}")
    print(f"    y_val: {y_val.shape}")
    
    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n  DataLoaders created:")
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def create_xhr_test_loader(data, norm_stats, batch_size=32):
    """
    Create test dataloader for G6sulfur inference.
    
    Args:
        data: dict from load_xhr_data()
        norm_stats: dict from load_normalization_stats_xhr()
        batch_size: batch size for DataLoader
    
    Returns:
        test_loader
    """
    print("\nCreating XHR test dataloader...")
    
    x_test = data['test_input'].values
    
    # Normalize using training stats
    x_test = normalize_zscore_pixel(
        x_test,
        norm_stats['input_detrend']['mean'],
        norm_stats['input_detrend']['std']
    )
    
    # Add channel dimension
    x_test = np.expand_dims(x_test, axis=1)
    
    print(f"  x_test: {x_test.shape}")
    
    # Convert to tensor
    x_test = torch.tensor(x_test, dtype=torch.float32)
    
    # Create dataloader (no targets for test)
    test_dataset = TensorDataset(x_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Test batches: {len(test_loader)}")
    
    return test_loader