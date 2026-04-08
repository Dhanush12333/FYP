# models/cbam_3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention3D(nn.Module):
    """3D Channel Attention Module (CAM)"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv3d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention3D(nn.Module):
    """3D Spatial Attention Module (SAM)"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention

class CBAM3D(nn.Module):
    """
    Convolutional Block Attention Module for 3D volumes
    Applies channel attention then spatial attention
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM3D, self).__init__()
        self.channel_attention = ChannelAttention3D(channels, reduction)
        self.spatial_attention = SpatialAttention3D(spatial_kernel)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# Optional: Residual CBAM block for better gradient flow
class ResidualCBAM3D(nn.Module):
    """Residual block with CBAM attention"""
    def __init__(self, channels, reduction=16):
        super(ResidualCBAM3D, self).__init__()
        self.cbam = CBAM3D(channels, reduction)
        
    def forward(self, x):
        return x + self.cbam(x)  # Residual connection