# main_xhr.py

import argparse
import os
import torch
from unet import UNet
from train import train_model
from utils_xhr import (
    set_seed,
    load_xhr_data,
    load_normalization_stats_xhr,
    get_latitude_values,
    create_xhr_dataloaders
)
from losses import compute_latitude_weights


def main():
    parser = argparse.ArgumentParser(description='Train XHR model for ERA5 downscaling')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=2000, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (smaller for XHR)')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model hyperparameters
    parser.add_argument('--initial_features', type=int, default=32, help='Initial UNet features')
    parser.add_argument('--depth', type=int, default=5, help='UNet depth')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--stats_path', type=str, default='data/norm_stats_xhr.pkl',
                        help='Path to normalization statistics')
    parser.add_argument('--save_path', type=str, default='ckpts/xhr_model.pth',
                        help='Model checkpoint save path')
    
    # Latitude weighting
    parser.add_argument('--no_lat_weights', dest='use_lat_weights', action='store_false',
                        help='Disable latitude-weighted MSE loss')
    parser.set_defaults(use_lat_weights=True)
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"XHR RESIDUAL PREDICTION TRAINING")
    print(f"{'='*80}")
    print(f"Task: CMIP6 (detrended) -> Residual (ERA5 - CMIP6_interp)")
    print(f"Resolution: 721 x 1440 (0.25 degree)")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"UNet depth: {args.depth}, initial features: {args.initial_features}")
    print(f"Latitude weighting: {args.use_lat_weights}")
    print(f"Save path: {args.save_path}")
    print(f"{'='*80}\n")
    
    # Load data
    data = load_xhr_data(args.data_dir)
    
    # Load normalization stats
    norm_stats = load_normalization_stats_xhr(args.stats_path)
    
    # Compute latitude weights if requested
    lat_weights = None
    if args.use_lat_weights:
        print("\nComputing latitude weights...")
        lat_values = get_latitude_values(data['train_input'])
        lat_weights = compute_latitude_weights(lat_values)
        print(f"  Latitude weights shape: {lat_weights.shape}")
        print(f"  Weight range: [{lat_weights.min():.4f}, {lat_weights.max():.4f}]")
    
    # Create dataloaders
    train_loader, val_loader = create_xhr_dataloaders(
        data=data,
        norm_stats=norm_stats,
        batch_size=args.batch_size
    )
    
    # Initialize model
    print(f"\nInitializing UNet model...")
    model = UNet(
        in_channels=1,
        out_channels=1,
        initial_features=args.initial_features,
        depth=args.depth,
        dropout=args.dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
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