import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Standard double convolution block for U-Net architecture."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetDCP(nn.Module):
    def __init__(self, patch_size = 7, gif_radius = 8, gif_eps = 1e-5, t_min_sky = 0.4, t_min_global = 0.3):
        super(UNetDCP, self).__init__()
        self.patch_size = patch_size
        self.gif_radius = gif_radius
        self.gif_eps = gif_eps
        self.t_min_sky = t_min_sky
        self.t_min_global = t_min_global

        #U-Net transmission sub-net
        ##Encoder
        self.dconv1 = DoubleConv(4,64)
        self.pool1 = nn.MaxPool2d(2)
        self.dconv2 = DoubleConv(64,128)
        self.pool2 = nn.MaxPool2d(2)
        
        #Bottleneck
        self.dconv4 = DoubleConv(128,256)

        #Decoder
        self.upconv1 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.dconv5 = DoubleConv(256,128)
        self.upconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.dconv6 = DoubleConv(128,64)

        #Final transmission output
        self.out_conv = nn.Conv2d(64,1,kernel_size=1)

        #Atmospheric light sub-net
        ##Downsampling path
        self.a_conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1)
        self.a_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.a_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        #Global atmospheric light
        self.a_pool = nn.AdaptiveAvgPool2d(1)
        self.a_fc1 = nn.Linear(64, 16)
        self.a_fc2 = nn.Linear(16, 3)
    
    def _compute_dark_channel(self, img):
        min_rgb, _ = torch.min(img, dim=1, keepdim=True)
        pad = self.patch_size // 2
        return -F.max_pool2d(-min_rgb, kernel_size=self.patch_size, stride=1, padding=pad)
    
    def guided_filter(self, I, p):
        ks = self.gif_radius * 2 + 1
        pad = self.gif_radius
        def box_filter(x):
            return F.avg_pool2d(x, kernel_size=ks, stride=1, padding=pad, count_include_pad=False)

        mean_I, mean_p = box_filter(I), box_filter(p)
        mean_Ip, mean_II = box_filter(I * p), box_filter(I * I)
        
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + self.gif_eps)
        b = mean_p - a * mean_I
        
        return box_filter(a) * I + box_filter(b)

    def forward(self, x):
        dc = self._compute_dark_channel(x)
        features = torch.cat([x, dc], dim=1)

        # Sky Branch
        sky_feat = F.relu(self.sky_conv1(features))
        sky_mask = torch.sigmoid(self.sky_conv2(sky_feat))

        # U-Net Transmission Branch
        d1 = self.dconv1(features)
        d2 = self.dconv2(self.pool1(d1))
        
        # Bottleneck
        b = self.dconv4(self.pool2(d2))
        
        # Decoder
        u1 = self.upconv1(b)
        if u1.shape[2:] != d2.shape[2:]:
            u1 = F.interpolate(u1, size=d2.shape[2:], mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, d2], dim=1) # Skip connection
        u1 = self.dconv5(u1)
        
        u2 = self.upconv2(u1)
        if u2.shape[2:] != d1.shape[2:]:
            u2 = F.interpolate(u2, size=d1.shape[2:], mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, d1], dim=1) # Skip connection
        u2 = self.dconv6(u2)
        
        t_raw = torch.sigmoid(self.out_conv(u2))

        # Differentiable Guided Filter Post-Processing
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        t_refined = self.guided_filter(gray, t_raw)

        # Sky Adaptive Correction
        t_min_sky = torch.tensor(self.t_min_sky, device=x.device)
        t_corrected = t_refined * (1 - sky_mask) + torch.max(t_refined, t_min_sky) * sky_mask
        t_final = torch.clamp(t_corrected, min=self.t_min_global)

        # Atmospheric Light Branch
        a_feat = F.relu(self.a_conv1(features))
        a_feat = F.relu(self.a_conv2(a_feat))
        a_feat = F.relu(self.a_conv3(a_feat))
        a_feat = self.a_pool(a_feat).view(x.size(0), -1)
        A = torch.sigmoid(self.a_fc2(F.relu(self.a_fc1(a_feat)))).view(-1, 3, 1, 1)

        # Physics Recovery
        clean = (x - A) / t_final + A
        clean = torch.clamp(clean, 0.0, 1.0)

        return clean, t_final, A
        