# main_residual.py

import argparse
import os
import torch
from unet import UNet
from train import train_model
from utils import *
from losses import compute_latitude_weights
import xarray as xr

def main():
    parser = argparse.ArgumentParser(description='Train ML model for residual prediction')
    
    # Training hyperparameters (same as original)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_path', type=str, default='ckpts/model.pth')
    
    # Data configuration
    parser.add_argument('--variables', type=str, nargs='+', default=['pr_hr'])
    parser.add_argument('--train_start', type=str, default='1850')
    parser.add_argument('--train_end', type=str, default='1980')
    parser.add_argument('--val_start', type=str, default='1981')
    parser.add_argument('--val_end', type=str, default='2000')
    
    # Residual-specific configuration
    parser.add_argument('--input_type', type=str, default='raw',
                        choices=['raw', 'detrend_gma', 'detrend_grid', 'detrend_gmt'],
                        help='Input preprocessing type')
    
    # Latitude weighting
    parser.add_argument('--no_lat_weights', dest='use_lat_weights', action='store_false')
    parser.set_defaults(use_lat_weights=True)
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_channels = len(args.variables)
    
    print(f"\n{'='*80}")
    print(f"RESIDUAL PREDICTION TRAINING")
    print(f"{'='*80}")
    print(f"Variables: {args.variables}")
    print(f"Input type: {args.input_type}")
    print(f"Target: Residual (HR - LR_interp)")
    print(f"Normalization: zscore_pixel")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Load residual dataset
    ds = xr.open_dataset(f"{args.data_dir}/MPI-ESM1-2-HR-LR_historical_residual_detrended.nc")
    
    # Compute latitude weights
    lat_weights = None
    if args.use_lat_weights:
        lat_values = get_latitude_values(ds)
        lat_weights = compute_latitude_weights(lat_values)
    
    # Load zscore_pixel stats for residual
    with open(f"{args.data_dir}/norm_stats_zscore_pixel_residual_detrended.pkl", 'rb') as f:
        norm_stats = pickle.load(f)
    
    # Create dataloaders using new function
    train_loader, val_loader = create_residual_dataloaders(
        ds=ds,
        variables=args.variables,
        norm_stats=norm_stats,
        train_period=(args.train_start, args.train_end),
        val_period=(args.val_start, args.val_end),
        batch_size=args.batch_size,
        input_type=args.input_type
    )
    
    # Initialize model
    model = UNet(
        in_channels=n_channels,
        out_channels=n_channels,
        initial_features=32,
        depth=5,
        dropout=0.2
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Create checkpoint directory
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Train model
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_path=args.save_path,
        n_epochs=args.epochs,
        patience=args.patience,
        lat_weights=lat_weights
    )
    
    print("\nTraining completed!")
    print(f"Model saved to: {args.save_path}")

if __name__ == "__main__":
    main()