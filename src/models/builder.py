"""
Model builder / registry.
Supports: MSFA-DeNet v2.
"""

import torch.nn as nn
from .msfa_denet_v2 import MSFADeNetV2


# ── Model Registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "msfa_denet_v2": MSFADeNetV2,
}


def build_model(config: dict) -> nn.Module:
    """
    Build a model from configuration.

    Config keys used:
        config["network"]["type"]:  name of the architecture (e.g. "msfa_denet_v2")

    Returns:
        An nn.Module model instance.
    """
    network_config = config.get("network", {})
    model_type = network_config.get("type", "msfa_denet_v2").lower()

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_type]

    # Forward architecture-specific kwargs from [network] config
    # (e.g., channels = 64 for MSFA-DeNet v2)
    model_kwargs = {k: v for k, v in network_config.items() if k != "type"}
    try:
        return model_class(**model_kwargs)
    except TypeError:
        # Model doesn't accept extra kwargs — instantiate with defaults
        return model_class()
