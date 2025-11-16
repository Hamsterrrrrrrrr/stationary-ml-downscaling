# losses.py

import torch
import torch.nn as nn
import numpy as np


class WeightedMSELoss(nn.Module):
    """
    MSE Loss weighted by latitude (cosine of latitude for area weighting)
    """
    def __init__(self, lat_weights=None):
        """
        Args:
            lat_weights: Precomputed latitude weights, shape [1, 1, H, 1]
                        If None, uses standard MSE loss
        """
        super(WeightedMSELoss, self).__init__()
        if lat_weights is not None:
            self.register_buffer('lat_weights', lat_weights)
        else:
            self.lat_weights = None
        
    def forward(self, output, target):
        """
        Args:
            output: Model predictions [B, C, H, W]
            target: Ground truth [B, C, H, W]
        
        Returns:
            Weighted MSE loss (scalar)
        """
        # Compute squared errors: [B, C, H, W]
        squared_errors = (output - target) ** 2
        
        if self.lat_weights is not None:
            # Apply latitude weights: [B, C, H, W] * [1, 1, H, 1] -> [B, C, H, W]
            weighted_errors = squared_errors * self.lat_weights
            
            # Total weight applied to all elements
            # lat_weights: [1, 1, H, 1]
            # When broadcasting over [B, C, H, W], total weight is:
            B, C, H, W = output.shape
            total_weight = self.lat_weights.sum() * B * C * W
            
            # Weighted mean
            loss = weighted_errors.sum() / total_weight
        else:
            loss = squared_errors.mean()
        
        return loss


def compute_latitude_weights(lat_values):
    """
    Compute cosine-based area weights for latitude values.
    
    Args:
        lat_values: 1D array of latitude values in degrees, shape (H,)
    
    Returns:
        weights: Tensor of shape [1, 1, H, 1] for broadcasting
    """
    # Convert to radians
    lat_rad = np.deg2rad(lat_values)
    
    # Cosine weighting (area of grid cells)
    weights = np.cos(lat_rad)
    
    # Normalize so mean weight = 1 (optional, for interpretability)
    weights = weights / weights.mean()
    
    # Reshape for broadcasting: [1, 1, H, 1]
    weights = torch.tensor(weights, dtype=torch.float32).view(1, 1, -1, 1)
    
    return weights