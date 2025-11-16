# main.py


# Run all 28 experiments (4 variables × 7 normalizations) in sequence

# /usr/local/anaconda3/bin/python main.py --variables tas_hr --normalization none --save_path ckpts/tas_none.pth && \
# /usr/local/anaconda3/bin/python main.py --variables tas_hr --normalization minmax_global --save_path ckpts/tas_minmax_global.pth && \
# /usr/local/anaconda3/bin/python main.py --variables tas_hr --normalization minmax_pixel --save_path ckpts/tas_minmax_pixel.pth && \
# /usr/local/anaconda3/bin/python main.py --variables tas_hr --normalization zscore_global --save_path ckpts/tas_zscore_global.pth && \
# /usr/local/anaconda3/bin/python main.py --variables tas_hr --normalization zscore_pixel --save_path ckpts/tas_zscore_pixel.pth && \
# /usr/local/anaconda3/bin/python main.py --variables tas_hr --normalization instance_zscore --save_path ckpts/tas_instance_zscore.pth && \
# /usr/local/anaconda3/bin/python main.py --variables tas_hr --normalization instance_minmax --save_path ckpts/tas_instance_minmax.pth && \
# /usr/local/anaconda3/bin/python main.py --variables pr_hr --normalization none --save_path ckpts/pr_none.pth && \
# /usr/local/anaconda3/bin/python main.py --variables pr_hr --normalization minmax_global --save_path ckpts/pr_minmax_global.pth && \
# /usr/local/anaconda3/bin/python main.py --variables pr_hr --normalization minmax_pixel --save_path ckpts/pr_minmax_pixel.pth && \
# /usr/local/anaconda3/bin/python main.py --variables pr_hr --normalization zscore_global --save_path ckpts/pr_zscore_global.pth && \
# /usr/local/anaconda3/bin/python main.py --variables pr_hr --normalization zscore_pixel --save_path ckpts/pr_zscore_pixel.pth && \
# /usr/local/anaconda3/bin/python main.py --variables pr_hr --normalization instance_zscore --save_path ckpts/pr_instance_zscore.pth && \
# /usr/local/anaconda3/bin/python main.py --variables pr_hr --normalization instance_minmax --save_path ckpts/pr_instance_minmax.pth && \
# /usr/local/anaconda3/bin/python main.py --variables hurs_hr --normalization none --save_path ckpts/hurs_none.pth && \
# /usr/local/anaconda3/bin/python main.py --variables hurs_hr --normalization minmax_global --save_path ckpts/hurs_minmax_global.pth && \
# /usr/local/anaconda3/bin/python main.py --variables hurs_hr --normalization minmax_pixel --save_path ckpts/hurs_minmax_pixel.pth && \
# /usr/local/anaconda3/bin/python main.py --variables hurs_hr --normalization zscore_global --save_path ckpts/hurs_zscore_global.pth && \
# /usr/local/anaconda3/bin/python main.py --variables hurs_hr --normalization zscore_pixel --save_path ckpts/hurs_zscore_pixel.pth && \
# /usr/local/anaconda3/bin/python main.py --variables hurs_hr --normalization instance_zscore --save_path ckpts/hurs_instance_zscore.pth && \
# /usr/local/anaconda3/bin/python main.py --variables hurs_hr --normalization instance_minmax --save_path ckpts/hurs_instance_minmax.pth && \
# /usr/local/anaconda3/bin/python main.py --variables sfcWind_hr --normalization none --save_path ckpts/sfcWind_none.pth && \
# /usr/local/anaconda3/bin/python main.py --variables sfcWind_hr --normalization minmax_global --save_path ckpts/sfcWind_minmax_global.pth && \
# /usr/local/anaconda3/bin/python main.py --variables sfcWind_hr --normalization minmax_pixel --save_path ckpts/sfcWind_minmax_pixel.pth && \
# /usr/local/anaconda3/bin/python main.py --variables sfcWind_hr --normalization zscore_global --save_path ckpts/sfcWind_zscore_global.pth && \
# /usr/local/anaconda3/bin/python main.py --variables sfcWind_hr --normalization zscore_pixel --save_path ckpts/sfcWind_zscore_pixel.pth && \
# /usr/local/anaconda3/bin/python main.py --variables sfcWind_hr --normalization instance_zscore --save_path ckpts/sfcWind_instance_zscore.pth && \
# /usr/local/anaconda3/bin/python main.py --variables sfcWind_hr --normalization instance_minmax --save_path ckpts/sfcWind_instance_minmax.pth


import argparse
import os
import torch
from unet import UNet
from train import train_model
from utils import *
from losses import compute_latitude_weights


def main():
    parser = argparse.ArgumentParser(description='Train ML model for climate downscaling')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=2000, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing data files')
    parser.add_argument('--stats_path', type=str, default='data/norm_stats.pkl', 
                        help='Path to normalization statistics pickle file')
    parser.add_argument('--save_path', type=str, default='ckpts/model.pth', 
                        help='Model checkpoint save path')
    
    # Data configuration
    parser.add_argument('--variables', type=str, nargs='+', default=['pr_hr'], 
                        help='Variables to use (e.g., pr_hr, tas_hr, hurs_hr, sfcWind_hr)')
    parser.add_argument('--train_start', type=str, default='1850', help='Training start year')
    parser.add_argument('--train_end', type=str, default='1980', help='Training end year')
    parser.add_argument('--val_start', type=str, default='1981', help='Validation start year')
    parser.add_argument('--val_end', type=str, default='2000', help='Validation end year')
    
    # Normalization method
    parser.add_argument('--normalization', type=str, default='minmax_global',
                        choices=['none', 'minmax_global', 'minmax_pixel', 'zscore_global', 
                                'zscore_pixel', 'instance_zscore', 'instance_minmax'],
                        help='Normalization method (none = no normalization)')
    
    # Latitude weighting (enabled by default)
    parser.add_argument('--no_lat_weights', dest='use_lat_weights', action='store_false',
                       help='Disable latitude-weighted MSE loss')
    parser.set_defaults(use_lat_weights=True)
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine number of channels
    n_channels = len(args.variables)
    
    print(f"\n{'='*80}")
    print(f"CLIMATE DOWNSCALING TRAINING")
    print(f"{'='*80}")
    print(f"Variables: {args.variables}")
    print(f"Number of channels: {n_channels}")
    print(f"Normalization: {args.normalization}")
    print(f"Stats path: {args.stats_path}") 
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Training period: {args.train_start}-{args.train_end}")
    print(f"Validation period: {args.val_start}-{args.val_end}")
    print(f"Latitude weighting: {args.use_lat_weights}")
    print(f"Save path: {args.save_path}") 
    print(f"{'='*80}\n")
    
    # Load historical dataset
    ds_hist = load_historical_dataset(args.data_dir)
    
    # Compute latitude weights if requested
    lat_weights = None
    if args.use_lat_weights:
        print("\nComputing latitude weights...")
        lat_values = get_latitude_values(ds_hist)
        lat_weights = compute_latitude_weights(lat_values)
        print(f"  Latitude weights shape: {lat_weights.shape}")
        print(f"  Weight range: [{lat_weights.min():.4f}, {lat_weights.max():.4f}]")
    
    # Load normalization statistics
    norm_stats = load_normalization_stats(args.stats_path)
    
    # Verify that stats exist for all requested variables
    print("\nVerifying normalization statistics...")
    for var in args.variables:
        var_base = var.replace('_hr', '')
        if var_base not in norm_stats:
            raise ValueError(f"Variable '{var_base}' not found in normalization stats!")
        print(f" Stats found for {var_base}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        ds=ds_hist,
        variables=args.variables,
        norm_stats=norm_stats,
        train_period=(args.train_start, args.train_end),
        val_period=(args.val_start, args.val_end),
        batch_size=args.batch_size,
        normalization=args.normalization
    )
    
    # Initialize model
    print(f"\nInitializing UNet model...")
    model = UNet(
        in_channels=n_channels,
        out_channels=n_channels,
        initial_features=32,
        depth=5,
        dropout=0.2
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Create checkpoint directory
    save_dir = os.path.dirname(args.save_path)
    if save_dir:  
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nCheckpoint directory: {save_dir}")
    
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")
    
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
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"Model saved to: {args.save_path}")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss: {val_losses[-1]:.6f}")
    print(f"Best val loss: {min(val_losses):.6f}")


if __name__ == "__main__":
    main()