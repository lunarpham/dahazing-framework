import torch
import torch.nn as nn

from .components import BReLU, Maxout

class DehazeNet(nn.Module):
    """
    A lightweight CNN model designed to estimate the transmission map t(x).
    Architecture inspired by the original DehazeNet paper.
    """
    def __init__(self):
        super(DehazeNet, self).__init__()
        
        # Layer 1: Feature Extraction
        # Extracts 16 feature maps
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        
        # Layer 2: Multi-scale Mapping
        # Parallel convolutions with different kernel sizes
        self.conv2_1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv2_3 = nn.Conv2d(16, 16, kernel_size=7, padding=3)
        
        # Layer 3: Local Extremum (Maxout)
        # Groups the 48 feature maps into 16 output maps by taking max across 3 pieces
        self.maxout = Maxout(in_channels=48, num_pieces=3)
        
        # Layer 4: Non-linear Regression
        # Predicts the 1-channel transmission map
        self.conv3 = nn.Conv2d(16, 1, kernel_size=7, padding=3)
        self.brelu = BReLU(t_max=1.0)
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # 1. Feature Extraction
        out = self.conv1(x)
        
        # 2. Multi-scale Mapping
        out1 = self.conv2_1(out)
        out2 = self.conv2_2(out)
        out3 = self.conv2_3(out)
        
        # Concatenate outputs
        out = torch.cat([out1, out2, out3], dim=1) # Shape: (B, 48, H, W)
        
        # 3. Local Extremum
        out = self.maxout(out) # Shape: (B, 16, H, W)
        
        # 4. Non-linear Regression
        out = self.conv3(out) # Shape: (B, 1, H, W)
        out = self.brelu(out)
        
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Prevent dying gradients: initialize the final convolution bias to 0.5
        # This ensures the output is well within BReLU [0, 1] range and > t0 (0.1).
        if hasattr(self, 'conv3') and self.conv3.bias is not None:
            nn.init.constant_(self.conv3.bias, 0.5)
