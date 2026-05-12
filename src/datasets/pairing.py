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
    Auto-pair RESIDE dataset images.
    RESIDE usually names hazy images as <id>_<params>.jpg and clear images as <id>.jpg
    Returns a list of tuples: (hazy_path, clear_path)
    """
    hazy_files = find_image_files(hazy_dir)
    clear_files = find_image_files(clear_dir)
    
    # Map clear files by their stem (e.g., '1400' from '1400.jpg')
    clear_map = {}
    for cf in clear_files:
        basename = os.path.basename(cf)
        stem = os.path.splitext(basename)[0]
        clear_map[stem] = cf
        
    pairs = []
    
    for hf in hazy_files:
        basename = os.path.basename(hf)
        stem = os.path.splitext(basename)[0]
        
        # the clear ID is usually the part before the first underscore in RESIDE
        clear_id = stem.split('_')[0]
        
        if clear_id in clear_map:
            pairs.append((hf, clear_map[clear_id]))
            
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
