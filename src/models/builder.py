"""
Model builder / registry for DehazeNet.
Add new architectures here as the project grows.
"""

import torch.nn as nn
from .dehazenet import DehazeNet
from .dehazenet_plus import DehazeNetPlus
from .dehazenet_direct import DehazeNetDirect
from .dehazenet_hybrid import DehazeNetHybrid
from .aodnet import AODNet
from .aodnet_enhanced import AODNetEnhanced
from .aodnet_pa import AODPANet
from .aodnet_capa import AODCAPANet


# ── Model Registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "dehazenet": DehazeNet,
    "dehazenet_plus": DehazeNetPlus,
    "dehazenet_direct": DehazeNetDirect,
    "dehazenet_hybrid": DehazeNetHybrid,
    "aodnet": AODNet,
    "aodnet_enhanced": AODNetEnhanced,
    "aodnet_pa": AODPANet,
    "aodnet_capa": AODCAPANet,
}

# Models that predict clean images directly (no transmission map → physics)
DIRECT_MODELS = {"dehazenet_direct", "dehazenet_hybrid", "aodnet", "aodnet_enhanced", "aodnet_pa", "aodnet_capa"}


def build_model(config: dict) -> nn.Module:
    """
    Build a model from configuration.

    Config keys used:
        config["network"]["type"]:  name of the architecture (e.g. "dehazenet")

    Returns:
        An nn.Module model instance.
    """
    network_config = config.get("network", {})
    model_type = network_config.get("type", "dehazenet").lower()

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_type]
    return model_class()
