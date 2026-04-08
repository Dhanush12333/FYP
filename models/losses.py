# models/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models

class DICELoss(nn.Module):
    """
    DICE Loss for 3D medical images
    Good for handling class imbalance in stenosis regions
    """
    def __init__(self, smooth=1e-6):
        super(DICELoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice

class CombinedDICELoss(nn.Module):
    """DICE loss that works on both volume and per-slice basis"""
    def __init__(self, smooth=1e-6, per_slice=False):
        super(CombinedDICELoss, self).__init__()
        self.smooth = smooth
        self.per_slice = per_slice
        
    def forward(self, pred, target):
        if self.per_slice:
            # Compute DICE per slice then average
            batch_size, channels, depth, h, w = pred.shape
            dice_sum = 0
            for b in range(batch_size):
                for d in range(depth):
                    pred_slice = pred[b, :, d, :, :].reshape(-1)
                    target_slice = target[b, :, d, :, :].reshape(-1)
                    intersection = (pred_slice * target_slice).sum()
                    dice = (2. * intersection + self.smooth) / (pred_slice.sum() + target_slice.sum() + self.smooth)
                    dice_sum += (1 - dice)
            return dice_sum / (batch_size * depth)
        else:
            # Volume-wise DICE
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            intersection = (pred_flat * target_flat).sum()
            dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
            return 1 - dice

class PerceptualLoss3D(nn.Module):
    """
    Perceptual loss using a pre-trained 3D CNN.
    Since pre-trained 3D models are rare, we use a 2D model (VGG16)
    applied slice-by-slice, or train a simple 3D feature extractor.
    """
    def __init__(self, feature_extractor='vgg16', layers=None, use_3d=False):
        super(PerceptualLoss3D, self).__init__()
        self.use_3d = use_3d
        
        if use_3d:
            # Option 1: Use a simple trainable 3D encoder (will learn during training)
            self.feature_extractor = self._build_3d_encoder()
        else:
            # Option 2: Use pretrained 2D VGG16 slice-by-slice (more common)
            vgg = tv_models.vgg16(pretrained=True).features
            self.feature_extractor = nn.Sequential(*list(vgg.children())[:16])  # Up to conv3_3
            self.feature_extractor.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
                
        # Default layers for perceptual loss
        if layers is None:
            self.layers = ['relu1_2', 'relu2_2', 'relu3_3'] if not use_3d else ['conv1', 'conv2', 'conv3']
        
        self.criterion = nn.L1Loss()
        
    def _build_3d_encoder(self):
        """Simple 3D encoder for feature extraction (trainable)"""
        return nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        
    def forward(self, generated, target):
        if self.use_3d:
            # 3D approach
            gen_features = self.feature_extractor(generated)
            target_features = self.feature_extractor(target)
            return self.criterion(gen_features, target_features)
        else:
            # 2D slice-by-slice approach
            batch_size, channels, depth, h, w = generated.shape
            total_loss = 0
            
            for b in range(batch_size):
                for d in range(depth):
                    gen_slice = generated[b, :, d, :, :].repeat(1, 3, 1, 1)  # Convert to 3-channel for VGG
                    target_slice = target[b, :, d, :, :].repeat(1, 3, 1, 1)
                    
                    gen_feat = self.feature_extractor(gen_slice)
                    target_feat = self.feature_extractor(target_slice)
                    total_loss += self.criterion(gen_feat, target_feat)
                    
            return total_loss / (batch_size * depth)

class CombinedGeneratorLoss(nn.Module):
    """
    Combines all losses for generator training
    Total = λ_adv * L_adv + λ_L1 * L_L1 + λ_dice * L_dice + λ_perceptual * L_perceptual
    """
    def __init__(self, lambda_adv=1.0, lambda_l1=100.0, lambda_dice=0.5, lambda_perceptual=0.1):
        super(CombinedGeneratorLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.lambda_l1 = lambda_l1
        self.lambda_dice = lambda_dice
        self.lambda_perceptual = lambda_perceptual
        
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.l1_criterion = nn.L1Loss()
        self.dice_criterion = CombinedDICELoss(smooth=1e-6, per_slice=True)
        self.perceptual_criterion = PerceptualLoss3D(use_3d=False)  # Can switch to True
        
    def forward(self, fake_vol, real_vol, pred_fake):
        """
        Args:
            fake_vol: Generated volume from generator
            real_vol: Ground truth volume
            pred_fake: Discriminator output on fake_vol
        """
        # Adversarial loss
        target_for_G = torch.full_like(pred_fake, 1.0, device=pred_fake.device)
        loss_adv = self.adv_criterion(pred_fake, target_for_G)
        
        # L1 reconstruction loss
        loss_l1 = self.l1_criterion(fake_vol, real_vol)
        
        # DICE loss (handles structural similarity)
        loss_dice = self.dice_criterion(fake_vol, real_vol)
        
        # Perceptual loss (improves visual quality)
        loss_perceptual = self.perceptual_criterion(fake_vol, real_vol)
        
        # Total
        total = (self.lambda_adv * loss_adv + 
                 self.lambda_l1 * loss_l1 + 
                 self.lambda_dice * loss_dice + 
                 self.lambda_perceptual * loss_perceptual)
        
        return total, loss_adv, loss_l1, loss_dice, loss_perceptual