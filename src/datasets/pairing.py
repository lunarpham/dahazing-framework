import os
import glob


def find_image_files(directory):
    """Recursively find all images in a directory."""
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    return sorted(files)


def get_image_pairs(hazy_dir, clear_dir, **kwargs):

    hazy_files = find_image_files(hazy_dir)
    clear_files = find_image_files(clear_dir)

    clear_by_stem = {}
    clear_by_prefix = {}
    for cf in clear_files:
        stem = os.path.splitext(os.path.basename(cf))[0]
        clear_by_stem[stem] = cf
        prefix = stem.split('_')[0]
        clear_by_prefix.setdefault(prefix, cf)

    pairs = []
    for hf in hazy_files:
        stem = os.path.splitext(os.path.basename(hf))[0]
        prefix = stem.split('_')[0]

        if prefix in clear_by_prefix:
            pairs.append((hf, clear_by_prefix[prefix]))
        elif stem in clear_by_stem:
            pairs.append((hf, clear_by_stem[stem]))

    return pairs
