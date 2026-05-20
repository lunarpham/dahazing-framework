"""
Domain-balanced weighted sampler for multi-domain dehazing datasets.

When training on mixed datasets (RESIDE + O-Haze + NH-Haze + 3R nighttime),
the dominant domain (RESIDE, typically 90%+) overwhelms minority domains.
This sampler assigns inverse-frequency weights so each domain gets equal
training time per epoch.

Domain detection uses filename patterns:
    - O-Haze:    contains 'ohaze' or 'O-Haze' (case-insensitive)
    - NH-Haze:   contains 'nhhaze' or 'NH-Haze' (case-insensitive)
    - 3R night:  contains 'nighttime' or 'lowlight' (case-insensitive)
    - RESIDE:    everything else (default)
"""

import os
import re
from torch.utils.data import WeightedRandomSampler


# ── Domain Detection ─────────────────────────────────────────────────────────

# Patterns for domain classification (checked against full file path)
DOMAIN_PATTERNS = {
    'ohaze':    re.compile(r'o[-_]?haze', re.IGNORECASE),
    'nhhaze':   re.compile(r'nh[-_]?haze', re.IGNORECASE),
    'nighttime': re.compile(r'nighttime|night[-_]?time|lowlight|low[-_]?light',
                            re.IGNORECASE),
}


def classify_domain(file_path: str) -> str:
    """
    Classify a hazy image file path into its source domain.

    Returns one of: 'ohaze', 'nhhaze', 'nighttime', 'reside' (default).
    """
    # Check against known patterns (path + filename)
    path_str = file_path.replace('\\', '/')
    for domain_name, pattern in DOMAIN_PATTERNS.items():
        if pattern.search(path_str):
            return domain_name
    return 'reside'  # default


def build_domain_balanced_sampler(dataset, replacement: bool = True):
    """
    Build a WeightedRandomSampler that balances across haze domains.

    Each domain gets equal probability of being sampled, regardless of
    how many images it contains. This prevents the dominant domain (RESIDE)
    from overwhelming minority domains during training.

    Args:
        dataset: DehazeDataset instance with .pairs attribute.
        replacement: Whether to sample with replacement (default True).
                     Set True for balanced sampling across unequal domains.

    Returns:
        WeightedRandomSampler instance, and a dict of domain counts for logging.
    """
    # Classify each sample
    domains = []
    for hazy_path, _ in dataset.pairs:
        domains.append(classify_domain(hazy_path))

    # Count samples per domain
    domain_counts = {}
    for d in domains:
        domain_counts[d] = domain_counts.get(d, 0) + 1

    # Compute inverse-frequency weight for each domain
    # Weight = 1 / count → rare domains get higher weight
    domain_weights = {d: 1.0 / count for d, count in domain_counts.items()}

    # Assign weight to each sample based on its domain
    sample_weights = [domain_weights[d] for d in domains]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=replacement,
    )

    return sampler, domain_counts
