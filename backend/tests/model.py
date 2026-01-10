import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block with instance normalization"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual

class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale features"""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])
        
    def forward(self, features):
        # features: list of feature maps from low to high resolution
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], 
                                            size=laterals[i - 1].shape[-2:],
                                            mode='nearest')
        
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outputs

class Generator(nn.Module):
    """DeblurGAN-v2 Generator with FPN backbone"""
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, num_residual_blocks=9):
        super().__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, 7),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down1 = self._downsample_block(base_channels, base_channels * 2)
        self.down2 = self._downsample_block(base_channels * 2, base_channels * 4)
        self.down3 = self._downsample_block(base_channels * 4, base_channels * 8)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(base_channels * 8) for _ in range(num_residual_blocks)]
        )
        
        # FPN for multi-scale features
        self.fpn = FPN([base_channels, base_channels * 2, base_channels * 4, base_channels * 8], 
                       base_channels * 2)
        
        # Upsampling with skip connections
        self.up1 = self._upsample_block(base_channels * 8 + base_channels * 2, base_channels * 4)
        self.up2 = self._upsample_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up3 = self._upsample_block(base_channels * 2 + base_channels * 2, base_channels)
        
        # Output layer
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, out_channels, 7),
            nn.Tanh()
        )
        
    def _downsample_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _upsample_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        x0 = self.initial(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        # Residual blocks
        x3 = self.residual_blocks(x3)
        
        # FPN features
        fpn_features = self.fpn([x0, x1, x2, x3])
        
        # Decoder with skip connections
        up = self.up1(torch.cat([x3, fpn_features[3]], dim=1))
        up = self.up2(torch.cat([up, fpn_features[2]], dim=1))
        up = self.up3(torch.cat([up, fpn_features[1]], dim=1))
        
        return self.output(up)

class Discriminator(nn.Module):
    """PatchGAN Discriminator"""
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        def discriminator_block(in_ch, out_ch, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, base_channels, normalize=False),
            *discriminator_block(base_channels, base_channels * 2),
            *discriminator_block(base_channels * 2, base_channels * 4),
            *discriminator_block(base_channels * 4, base_channels * 8),
            nn.Conv2d(base_channels * 8, 1, 4, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features"""
    def __init__(self):
        super().__init__()
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True).features
        self.layers = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:18], # relu3_4
            vgg[18:27] # relu4_4
        ])
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.weights = [1.0, 1.0, 1.0, 1.0]
        
    def forward(self, pred, target):
        loss = 0
        x = pred
        y = target
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            y = layer(y)
            loss += self.weights[i] * F.l1_loss(x, y)
        
        return loss

def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)