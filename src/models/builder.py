"""
Model builder / registry for DehazeNet.
Supports: DehazeNet, AOD-Net, MSFA-Net, MSFA-Net Lite.
"""

import torch.nn as nn
from .dehazenet import DehazeNet
from .aodnet import AODNet
from .msfa_net import MSFANet
from .msfa_net_lite import MSFANetLite
from .dcpnet import DCPNet
from .unetdcp import UNetDCP


# ── Model Registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "dehazenet": DehazeNet,
    "aodnet": AODNet,
    "msfa_net": MSFANet,
    "msfa_net_lite": MSFANetLite,
    "dcpnet": DCPNet,
    "unetdcp": UNetDCP,
}

# Models that predict clean images directly (no transmission map → physics)
DIRECT_MODELS = {"aodnet", "msfa_net", "msfa_net_lite", "dcpnet", "unetdcp"}


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

    # Forward architecture-specific kwargs from [network] config
    # (e.g., channels = 64 for MSFA-Net)
    model_kwargs = {k: v for k, v in network_config.items() if k != "type"}
    try:
        return model_class(**model_kwargs)
    except TypeError:
        # Model doesn't accept extra kwargs — instantiate with defaults
        return model_class()
