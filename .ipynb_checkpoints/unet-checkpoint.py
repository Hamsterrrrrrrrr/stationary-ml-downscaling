import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, initial_features=32, depth=5, dropout=0.2):
        """
        UNet architecture for image-to-image translation.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            initial_features: Number of features in first encoder block
            depth: Number of encoder/decoder blocks
            dropout: Dropout probability
        """
        super(UNet, self).__init__()
        self.depth = depth
        self.dropout = dropout
        
        # Encoder
        self.enc_blocks = nn.ModuleList()
        current_channels = in_channels
        features = initial_features
        
        for _ in range(depth):
            self.enc_blocks.append(nn.Sequential(
                nn.Conv2d(current_channels, features, kernel_size=3, padding=1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ))
            current_channels = features
            features *= 2
            
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(current_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        # Decoder
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        
        for _ in reversed(range(depth)):
            features //= 2
            self.up_convs.append(
                nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
            )
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(features * 2, features, kernel_size=3, padding=1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ))
        
        # Final output convolution
        self.final_conv = nn.Conv2d(initial_features, out_channels, kernel_size=1)
    
    
    def forward(self, x):
        """
        Forward pass through UNet.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]
        
        Returns:
            Output tensor of shape [batch_size, out_channels, height, width]
        """
        # Store skip connections
        skips = []
        
        # Encoder path - downsampling
        for enc_block in self.enc_blocks:
            x = enc_block(x)
            skips.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path - upsampling
        for idx, (up_conv, dec_block) in enumerate(zip(self.up_convs, self.dec_blocks)):
            # Upsample
            x = up_conv(x)
            
            # Get corresponding skip connection
            skip = skips[-(idx + 1)]
            
            # Handle potential size mismatches due to odd input dimensions
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, 
                    size=skip.shape[2:], 
                    mode='bilinear', 
                    align_corners=True
                )
            
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            
            # Apply decoder block
            x = dec_block(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x

