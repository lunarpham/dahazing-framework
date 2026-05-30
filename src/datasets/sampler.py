"""Domain-balanced weighted sampler for multi-domain dehazing datasets."""

import os
import re
from torch.utils.data import WeightedRandomSampler


# Domain detection patterns (checked against full file path)
DOMAIN_PATTERNS = {
    'ohaze':    re.compile(r'o[-_]?haze', re.IGNORECASE),
    'nhhaze':   re.compile(r'nh[-_]?haze', re.IGNORECASE),
    'nighttime': re.compile(r'nighttime|night[-_]?time|lowlight|low[-_]?light',
                            re.IGNORECASE),
}


def classify_domain(file_path: str) -> str:
    """Classify a file path into its source domain. Default: 'reside'."""
    path_str = file_path.replace('\\', '/')
    for domain_name, pattern in DOMAIN_PATTERNS.items():
        if pattern.search(path_str):
            return domain_name
    return 'reside'


def build_domain_balanced_sampler(dataset, replacement: bool = True):
    """Build a WeightedRandomSampler that balances across haze domains.

    Returns:
        (WeightedRandomSampler, dict of domain counts)
    """
    domains = [classify_domain(hazy_path) for hazy_path, _ in dataset.pairs]

    domain_counts = {}
    for d in domains:
        domain_counts[d] = domain_counts.get(d, 0) + 1

    # Inverse-frequency weights: rare domains get higher weight
    domain_weights = {d: 1.0 / count for d, count in domain_counts.items()}
    sample_weights = [domain_weights[d] for d in domains]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=replacement,
    )

    return sampler, domain_counts
