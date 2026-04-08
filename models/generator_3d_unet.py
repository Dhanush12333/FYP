# models/generator_3d_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 norm='inst'):
        super().__init__()
        self.num_levels = num_levels
        self.preserve_depth = preserve_depth

        self.enc_blocks = nn.ModuleList()
        in_ch = in_channels
        filters = base_filters
        for lvl in range(num_levels):
            self.enc_blocks.append(Conv3DBlock(in_ch, filters, norm=norm))
            in_ch = filters
            filters *= 2

        self.bottleneck = Conv3DBlock(in_ch, filters, norm=norm)

        pool_kernel = (1,2,2) if preserve_depth else 2
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel)

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
        for enc in self.enc_blocks:
            cur = enc(cur)
            skips.append(cur)
            cur = self.pool(cur)

        cur = self.bottleneck(cur)

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
