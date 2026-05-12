import torch

def recover_image(hazy_img, t_map, atm_light, t0=0.1):
    """
    Recover the clear image using Koschmieder's law.
    J(x) = (I(x) - A) / max(t(x), t0) + A
    
    Args:
        hazy_img: Tensor (B, C, H, W) in range [0, 1]
        t_map: Estimated transmission map Tensor (B, 1, H, W)
        atm_light: Estimated atmospheric light Tensor (B, C, 1, 1) or scalar
        t0: Lower bound for transmission to prevent division by zero
        
    Returns:
        J: Dehazed Image Tensor (B, C, H, W) in range [0, 1]
    """
    # Ensure t_map never goes below t0
    t_clamped = torch.max(t_map, torch.tensor(t0, device=t_map.device, dtype=t_map.dtype))
    
    # Apply Koschmieder's law
    J = (hazy_img - atm_light) / t_clamped + atm_light
    
    # Clip values to valid image range
    J = torch.clamp(J, 0.0, 1.0)
    return J
