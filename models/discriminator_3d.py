# discriminator_3d.py
import torch
import torch.nn as nn

class Conv3dBlock(nn.Module):
    """Conv3d -> (InstanceNorm3d or BatchNorm3d) -> LeakyReLU (non-inplace for safety)"""
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1, norm='inst'):
        super().__init__()
        layers = [nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]
        if norm == 'bn':
            layers.append(nn.BatchNorm3d(out_ch))
        else:
            # affine=False -> no learnable scale/shift (safer for double backward)
            layers.append(nn.InstanceNorm3d(out_ch, affine=False))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Discriminator3D(nn.Module):
    """
    3D PatchGAN discriminator.
    Input:  [B, in_channels, D, H, W]
    Output: [B, 1, D', H', W'] (logits)
    """

    def __init__(self, in_channels=1, base_filters=64, n_layers=4, preserve_depth=True, norm='inst'):
        super().__init__()
        self.preserve_depth = preserve_depth
        self.norm = norm

        # First layer
        if preserve_depth:
            self.layer0 = nn.Sequential(
                nn.Conv3d(in_channels, base_filters,
                          kernel_size=(3, 4, 4),
                          stride=(1, 2, 2),
                          padding=(1, 1, 1),
                          bias=True),
                nn.LeakyReLU(0.2, inplace=False)
            )
        else:
            self.layer0 = nn.Sequential(
                nn.Conv3d(in_channels, base_filters,
                          kernel_size=4, stride=2, padding=1, bias=True),
                nn.LeakyReLU(0.2, inplace=False)
            )

        # Build body
        layers = []
        nf = base_filters
        for i in range(1, n_layers):
            in_nf = nf
            nf = min(nf * 2, base_filters * 8)
            if preserve_depth:
                layers.append(Conv3dBlock(in_nf, nf,
                                          kernel_size=(3, 4, 4),
                                          stride=(1, 2, 2),
                                          padding=(1, 1, 1),
                                          norm=norm))
            else:
                layers.append(Conv3dBlock(in_nf, nf,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          norm=norm))
        self.body = nn.Sequential(*layers)

        # Final classifier conv
        if preserve_depth:
            self.final = nn.Conv3d(nf, 1,
                                   kernel_size=(3, 4, 4),
                                   stride=(1, 1, 1),
                                   padding=(1, 1, 1),
                                   bias=True)
        else:
            self.final = nn.Conv3d(nf, 1,
                                   kernel_size=4,
                                   stride=1,
                                   padding=1,
                                   bias=True)

        self._init_weights()

    def forward(self, x):
        x = self.layer0(x)
        x = self.body(x)
        x = self.final(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.InstanceNorm3d, nn.BatchNorm3d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.constant_(m.weight, 1.0)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0.0)


if __name__ == "__main__":
    print("Running Discriminator3D smoke test...")
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = Discriminator3D(in_channels=1, base_filters=32, n_layers=4, preserve_depth=True).to(device)
    x = torch.randn(1, 1, 16, 256, 256, device=device, requires_grad=True)

    out = D(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    out_mean = out.mean()
    out_mean.backward()  # test backward pass
    print("Backward pass OK ✅")
