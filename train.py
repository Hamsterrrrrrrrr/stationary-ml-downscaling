import torch
import torch.nn as nn
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses import WeightedMSELoss 

def train_model(model, train_loader, val_loader,
                optimizer, device,  
                save_path='ckpts/model.pth',
                n_epochs=2000,
                patience=30,
                lat_weights=None):


    model.to(device)
    
    if lat_weights is not None:
        lat_weights = lat_weights.to(device)
        mse_loss_fn = WeightedMSELoss(lat_weights)
        print("Using latitude-weighted MSE loss")
    else:
        mse_loss_fn = nn.MSELoss()
        print("Using standard MSE loss")
        
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Variables for early stopping and loss tracking
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Checkpoint loading 
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print(f"Resuming from epoch {start_epoch} with best_val_loss = {best_val_loss:.3e}")
    else:
        start_epoch = 0
        print("Starting training from scratch")

    # Training loop
    for epoch in range(start_epoch, n_epochs):
        
        start_time = time.time()
        model.train()
        train_running_loss = 0.0

        # Training phase
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):  
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Data sanity check
            if epoch == 0 and batch_idx == 0:
                print(f"\n{'='*60}")
                print("DATA RANGE CHECK")
                print(f"{'='*60}")
                print(f"Input (batch_x):")
                print(f"  Shape: {batch_x.shape}")
                print(f"  Min: {batch_x.min():.4f}, Max: {batch_x.max():.4f}")
                print(f"  Mean: {batch_x.mean():.4f}, Std: {batch_x.std():.4f}")
                print(f"Target (batch_y):")
                print(f"  Shape: {batch_y.shape}")
                print(f"  Min: {batch_y.min():.4f}, Max: {batch_y.max():.4f}")
                print(f"  Mean: {batch_y.mean():.4f}, Std: {batch_y.std():.4f}")
                print(f"{'='*60}\n")
            
            # Forward pass
            outputs = model(batch_x)
            loss = mse_loss_fn(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item() * batch_x.size(0)
            
        # Calculate average training loss for the epoch
        epoch_loss = train_running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = mse_loss_fn(outputs, batch_y)
                val_running_loss += loss.item() * batch_x.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {epoch_loss:.2e}, "
              f"Val Loss: {val_loss:.2e}, LR: {optimizer.param_groups[0]['lr']:.2e}, "
              f"Time: {epoch_duration:.2f}s")

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            torch.save(checkpoint, save_path)
            print(f"Best model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
    
    return train_losses, val_losses