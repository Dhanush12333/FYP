# models/generator_3d_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cbam_3d import CBAM3D, ResidualCBAM3D  # NEW import

class Conv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm='inst'):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        ]
        if norm == 'bn':
            layers.append(nn.BatchNorm3d(out_ch))
        else:
            layers.append(nn.InstanceNorm3d(out_ch, affine=True))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False))
        if norm == 'bn':
            layers.append(nn.BatchNorm3d(out_ch))
        else:
            layers.append(nn.InstanceNorm3d(out_ch, affine=True))
        layers.append(nn.ReLU(inplace=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Up3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, preserve_depth=True):
        super().__init__()
        if preserve_depth:
            self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(1,2,2), stride=(1,2,2))
        else:
            self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = Conv3DBlock(in_ch=out_ch*2, out_ch=out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            diff_d = skip.size(2) - x.size(2)
            diff_h = skip.size(3) - x.size(3)
            diff_w = skip.size(4) - x.size(4)
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2,
                          diff_d // 2, diff_d - diff_d // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

class Generator3D_UNet(nn.Module):
    def __init__(self,
                 in_channels=2,
                 out_channels=1,
                 base_filters=32,
                 num_levels=4,
                 preserve_depth=True,
                 norm='inst',
                 use_cbam=True,           # NEW: Enable/disable CBAM
                 cbam_position='both'):    # NEW: 'bottleneck', 'skip', or 'both'
        super().__init__()
        self.num_levels = num_levels
        self.preserve_depth = preserve_depth
        self.use_cbam = use_cbam
        self.cbam_position = cbam_position
        
        # Encoder blocks
        self.enc_blocks = nn.ModuleList()
        in_ch = in_channels
        filters = base_filters
        for lvl in range(num_levels):
            self.enc_blocks.append(Conv3DBlock(in_ch, filters, norm=norm))
            in_ch = filters
            filters *= 2

        # Bottleneck
        self.bottleneck = Conv3DBlock(in_ch, filters, norm=norm)
        
        # NEW: Add CBAM after bottleneck if enabled
        if use_cbam and cbam_position in ['bottleneck', 'both']:
            self.bottleneck_cbam = CBAM3D(filters, reduction=16)
        else:
            self.bottleneck_cbam = nn.Identity()

        # Pooling
        pool_kernel = (1,2,2) if preserve_depth else 2
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel)

        # NEW: Create CBAM modules for skip connections
        self.skip_cbams = nn.ModuleList()
        if use_cbam and cbam_position in ['skip', 'both']:
            skip_filters = [base_filters * (2**i) for i in range(num_levels)]
            for f in skip_filters:
                self.skip_cbams.append(CBAM3D(f, reduction=16))
        else:
            # Placeholder identity modules (no attention)
            for _ in range(num_levels):
                self.skip_cbams.append(nn.Identity())

        # Decoder blocks
        self.up_blocks = nn.ModuleList()
        filters = filters // 2
        for lvl in range(num_levels):
            self.up_blocks.append(Up3DBlock(in_ch=filters*2, out_ch=filters, preserve_depth=preserve_depth))
            filters = filters // 2

        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)
        self._init_weights()

    def forward(self, x):
        skips = []
        cur = x
        
        # Encoder path with CBAM on skip connections
        for i, enc in enumerate(self.enc_blocks):
            cur = enc(cur)
            # NEW: Apply CBAM to skip connection before pooling
            skip_with_cbam = self.skip_cbams[i](cur)
            skips.append(skip_with_cbam)
            cur = self.pool(cur)

        # Bottleneck with CBAM
        cur = self.bottleneck(cur)
        cur = self.bottleneck_cbam(cur)  # NEW: Apply CBAM after bottleneck

        # Decoder path
        for up, skip in zip(self.up_blocks, reversed(skips)):
            cur = up(cur, skip)

        out = self.final_conv(cur)
        # Ensure outputs are in 0..1 to match dataset normalization
        out = torch.sigmoid(out)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.InstanceNorm3d, nn.BatchNorm3d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
