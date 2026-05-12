import torch
import torch.nn as nn

class BReLU(nn.Module):
    """
    Bilateral ReLU (BReLU) limits the activation value in the range [0, t_max].
    For image transmission map, t_max is usually 1.0.
    """
    def __init__(self, t_max=1.0):
        super(BReLU, self).__init__()
        self.t_max = t_max

    def forward(self, x):
        return torch.clamp(x, 0.0, self.t_max)

class Maxout(nn.Module):
    """
    Spatial/Channel Maxout block used in DehazeNet to extract local extremum.
    Takes the maximum across `num_pieces` feature maps.
    """
    def __init__(self, in_channels, num_pieces):
        super(Maxout, self).__init__()
        self.in_channels = in_channels
        self.num_pieces = num_pieces
        self.out_channels = in_channels // num_pieces

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.size()
        # reshape to (B, C//num_pieces, num_pieces, H, W)
        x = x.view(B, self.out_channels, self.num_pieces, H, W)
        # take the max across the pieces
        x, _ = torch.max(x, dim=2)
        return x
