import os
import glob

def find_image_files(directory):
    """Recursively find all images in a directory."""
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    return sorted(files)

def auto_pair_reside(hazy_dir, clear_dir):
    """
    Auto-pair images from merged dehazing datasets.

    Supports multiple naming conventions found in common datasets:
      1. Exact match:  hazy/1234.jpg          ↔  clear/1234.jpg
      2. _hazy/_GT:    hazy/01_hazy.png        ↔  clear/01_GT.png
      3. RESIDE ITS:   hazy/0023_0.85_0.12.jpg ↔  clear/0023.png
      4. Suffixed clear files (any suffix):
                       hazy/0001_NighttimeHazy_1.jpg ↔  clear/0001_<anything>.jpg
         Clear files whose stem starts with a purely numeric prefix are ALSO
         indexed under that bare numeric ID. The suffix after the first
         underscore is ignored for matching purposes — it exists only for
         the user's own dataset organisation.

    Matching is tried in the order above; the first hit wins.

    Returns a list of tuples: (hazy_path, clear_path)
    """
    hazy_files = find_image_files(hazy_dir)
    clear_files = find_image_files(clear_dir)

    # Build the clear lookup map, indexed by stem.
    # Strategy 4: for any clear file whose stem begins with a purely numeric
    # prefix (e.g. "0001_nightimehazy" → prefix "0001"), also register it
    # under that bare ID.  setdefault ensures a plain "0001.jpg" always wins
    # over a suffixed variant if both happen to exist.
    clear_map = {}
    for cf in clear_files:
        stem = os.path.splitext(os.path.basename(cf))[0]
        clear_map[stem] = cf                        # full stem (strategies 1-2)

        if '_' in stem:
            prefix = stem.split('_')[0]
            if prefix.isdigit():
                clear_map.setdefault(prefix, cf)    # bare numeric ID (strategy 4)

    pairs = []

    for hf in hazy_files:
        stem = os.path.splitext(os.path.basename(hf))[0]

        # Strategy 1: exact stem match
        if stem in clear_map:
            pairs.append((hf, clear_map[stem]))
            continue

        # Strategy 2: _hazy → _GT suffix swap
        if stem.endswith('_hazy'):
            gt_stem = stem[:-5] + '_GT'
            if gt_stem in clear_map:
                pairs.append((hf, clear_map[gt_stem]))
                continue

        # Strategy 3 + 4: numeric prefix before the first underscore.
        # Covers RESIDE ITS (0023_0.85_0.12) and custom variants
        # (0001_NighttimeHazy_1, 0001_lowLight_1).
        if '_' in stem:
            prefix = stem.split('_')[0]
            if prefix.isdigit() and prefix in clear_map:
                pairs.append((hf, clear_map[prefix]))
                continue

    return pairs



def auto_pair_strict(hazy_dir, clear_dir):
    """
    Auto-pair images assuming they have exactly the same name or relative path.
    """
    hazy_files = find_image_files(hazy_dir)
    pairs = []
    
    for hf in hazy_files:
        rel_path = os.path.relpath(hf, hazy_dir)
        matching_clear = os.path.join(clear_dir, rel_path)
        if os.path.exists(matching_clear):
            pairs.append((hf, matching_clear))
            
    return pairs

def get_image_pairs(hazy_dir, clear_dir, strategy='reside'):
    """
    Get (hazy, clear) image path pairs using the specified strategy.
    Strategies: 'reside' (matches based on prefix), 'strict' (exact filename match)
    """
    if strategy == 'reside':
        return auto_pair_reside(hazy_dir, clear_dir)
    elif strategy == 'strict':
        return auto_pair_strict(hazy_dir, clear_dir)
    else:
        raise ValueError(f"Unknown pairing strategy: {strategy}")
