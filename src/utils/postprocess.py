"""
Post-processing utilities for dehazed images.
Applies local contrast enhancement to recover detail lost during reconstruction.
"""

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def clahe_enhance(img_np, clip_limit=2.0, grid_size=8):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to the luminance channel of a dehazed image.

    This recovers local contrast that is often lost when dehazing
    thick-haze images via Koschmieder's law.

    Args:
        img_np: NumPy array (H, W, 3) in range [0, 1], RGB format.
        clip_limit: CLAHE contrast clip limit (higher = more contrast).
        grid_size: Tile grid size for CLAHE (smaller = more local).

    Returns:
        Enhanced image as NumPy array (H, W, 3) in range [0, 1].
    """
    if not HAS_CV2:
        # Graceful fallback: return input unchanged if cv2 not available
        return img_np

    img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

    # Convert to LAB color space — only enhance luminance (L channel)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size)
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return result.astype(np.float32) / 255.0


def gamma_correction(img_np, gamma=0.85):
    """
    Apply gamma correction to adjust overall brightness/contrast.

    Args:
        img_np: NumPy array (H, W, 3) in range [0, 1].
        gamma: Gamma value. < 1 = brighter, > 1 = darker.

    Returns:
        Gamma-corrected image in range [0, 1].
    """
    return np.clip(np.power(img_np, gamma), 0.0, 1.0).astype(np.float32)


def post_process(img_np, enable_clahe=True, clip_limit=2.0, grid_size=8,
                 enable_gamma=False, gamma=0.85):
    """
    Full post-processing pipeline for dehazed images.

    Args:
        img_np: NumPy array (H, W, 3) in range [0, 1], RGB format.
        enable_clahe: Whether to apply CLAHE enhancement.
        clip_limit: CLAHE clip limit.
        grid_size: CLAHE tile grid size.
        enable_gamma: Whether to apply gamma correction.
        gamma: Gamma value for correction.

    Returns:
        Post-processed image as NumPy array (H, W, 3) in range [0, 1].
    """
    result = img_np.copy()

    if enable_clahe:
        result = clahe_enhance(result, clip_limit=clip_limit, grid_size=grid_size)

    if enable_gamma:
        result = gamma_correction(result, gamma=gamma)

    return result
