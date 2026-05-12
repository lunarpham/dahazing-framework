import torch
import torch.nn.functional as F

def get_dark_channel(img, window_size=15):
    """
    Calculate the dark channel of an image.
    img form: tensor (B, C, H, W)
    """
    # Min over channels
    min_pool = torch.min(img, dim=1, keepdim=True)[0]
    
    # Min over spatial patch
    pad = window_size // 2
    dark_channel = F.max_pool2d(
        (1 - min_pool), # Invert because max_pool finds max, we want min
        kernel_size=window_size,
        stride=1,
        padding=pad
    )
    return 1 - dark_channel

def estimate_atmospheric_light(img, dark_channel, top_percent=0.001):
    """
    Estimate Atmospheric Light A from the original image and its dark channel.
    img: (B, 3, H, W)
    dark_channel: (B, 1, H, W)
    """
    B, C, H, W = img.shape
    num_pixels = H * W
    num_top = max(1, int(num_pixels * top_percent))

    # Flatten the tensors for easier sorting
    dc_flat = dark_channel.view(B, -1)
    img_flat = img.view(B, C, -1)
    
    atm_lights = torch.zeros((B, C, 1, 1), device=img.device, dtype=img.dtype)
    
    # Process each image in the batch
    for b in range(B):
        # Find indices of the top top_percent brightest pixels in dark channel
        _, indices = torch.topk(dc_flat[b], num_top)
        
        # Get the corresponding pixels from the original image
        top_pixels = img_flat[b, :, indices]  # Shape: (C, num_top)
        
        # Robust estimation: use median of top pixels instead of max.
        # Max is noisy in thick haze (picks outlier bright pixels);
        # median is much more stable while still capturing airlight.
        if num_top == 1:
            a_robust = top_pixels[:, 0]
        else:
            a_robust = torch.median(top_pixels, dim=-1).values
        atm_lights[b, :, 0, 0] = a_robust
        
    return atm_lights
