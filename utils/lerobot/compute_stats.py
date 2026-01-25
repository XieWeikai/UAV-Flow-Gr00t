import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import tqdm
from PIL import Image
import einops
from datasets import load_dataset 

# Local import
import cv2

def load_image_as_numpy(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def compute_stats(
    dataset,
    batch_size: int = 16,
    num_workers: int = 4,
    max_num_samples: Optional[int] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute statistics (min, max, mean, std) for all float32/float64/image features.
    """
    features = dataset.features
    
    # Identify keys
    keys = []
    for key, feat in features.items():
        if key == "index": continue
        # HuggingFace Datasets features
        if hasattr(feat, "dtype") and feat.dtype in ["float32", "float64"]:
            keys.append(key)
        # Custom simplified check for "VideoFrame" or just dictionary check
        elif isinstance(feat, dict) and feat.get("_type") in ["Image", "VideoFrame"]:
             keys.append(key)
        # Check if it's a sequence of floats
        elif hasattr(feat, "feature") and hasattr(feat.feature, "dtype") and feat.feature.dtype in ["float32", "float64"]:
            keys.append(key)

    if not keys:
        return {}

    logging.info(f"Computing stats for keys: {keys}")

    # initialize
    sums = {k: None for k in keys}
    sq_sums = {k: None for k in keys}
    mins = {k: None for k in keys}
    maxs = {k: None for k in keys}
    counts = {k: 0 for k in keys}

    def process_batch(batch):
        batch_stats = {}
        for key in keys:
            data = batch[key]
            val = None
            
            # Handle list of images (dictionaries with path)
            if isinstance(data[0], dict) and "path" in data[0]:
                # Load images
                imgs = []
                for item in data:
                    p = item["path"]
                    img = load_image_as_numpy(p)
                    imgs.append(img)
                val = torch.from_numpy(np.stack(imgs)).float()
                # (B, H, W, C) -> (B, C, H, W)
                val = einops.rearrange(val, "b h w c -> b c h w")
                val = val / 255.0 # Normalize 0-1
                
            # Handle list of numbers
            elif isinstance(data[0], list): # Sequence
                 val = torch.tensor(data).float()
            elif isinstance(data[0], (float, int)):
                 val = torch.tensor(data).float()
                 if val.ndim == 1: val = val.unsqueeze(1)
            
            if val is not None:
                if val.ndim == 4: # Image (B, C, H, W)
                    reduce_dims = [0, 2, 3] # B, H, W
                    cur_count = val.shape[0] * val.shape[2] * val.shape[3]
                else: 
                    reduce_dims = [0] # Just B
                    cur_count = val.shape[0]
                
                b_min = val.amin(dim=reduce_dims)
                b_max = val.amax(dim=reduce_dims)
                b_sum = val.sum(dim=reduce_dims)
                b_sq_sum = (val ** 2).sum(dim=reduce_dims)
                
                batch_stats[key] = (b_min, b_max, b_sum, b_sq_sum, cur_count)
        return batch_stats

    num_samples = len(dataset)
    if max_num_samples is not None:
        num_samples = min(num_samples, max_num_samples)
    
    # We can iterate using dataset.iter(batch_size=batch_size)
    # Note: datasets.iter with batch_size returns a dict of lists
    
    iterator = dataset.iter(batch_size=batch_size)
    
    for i, batch in tqdm.tqdm(enumerate(iterator), total=num_samples // batch_size):
        if max_num_samples and i * batch_size >= max_num_samples:
            break
            
        b_stats = process_batch(batch)
        
        for k, (b_min, b_max, b_sum, b_sq_sum, b_count) in b_stats.items():
            if sums[k] is None:
                sums[k] = b_sum
                sq_sums[k] = b_sq_sum
                mins[k] = b_min
                maxs[k] = b_max
                counts[k] = b_count
            else:
                sums[k] += b_sum
                sq_sums[k] += b_sq_sum
                mins[k] = torch.minimum(mins[k], b_min)
                maxs[k] = torch.maximum(maxs[k], b_max)
                counts[k] += b_count

    # Finalize
    final_stats = {}
    for k in keys:
        if counts[k] > 0:
            mean = sums[k] / counts[k]
            var = (sq_sums[k] / counts[k]) - (mean ** 2)
            var = torch.clamp(var, min=0)
            std = torch.sqrt(var)
            final_stats[k] = {
                "min": mins[k],
                "max": maxs[k],
                "mean": mean,
                "std": std
            }
            
    return final_stats

def compute_episode_stats(episode_data, features):
    """
    Compute statistics for a single episode.
    episode_data: dict of values (lists or arrays or list of paths for images)
    features: dict of feature specs
    
    Returns: dict of stats {key: {mean, std, min, max}}
    """
    stats = {}
    for key, spec in features.items():
        if key == "index": continue
        
        dtype = spec.get("dtype")
        if not dtype and "dtype" in spec.get("feature", {}):
            dtype = spec["feature"]["dtype"]
            
        if dtype not in ["float32", "float64", "image", "video"]:
             continue
        
        if key not in episode_data:
            continue
            
        data = episode_data[key]
        
        val = None
        # Handle Images
        if dtype == "image" or dtype == "video":
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
                 imgs = [load_image_as_numpy(p) for p in data]
                 val = np.stack(imgs) # (T, H, W, C)
                 val = val.astype(np.float32) / 255.0
                 val = einops.rearrange(val, "t h w c -> t c h w")
            
        else:
             # Float data
             val = np.array(data)
             if val.ndim == 1:
                 val = val[:, None] # (T, 1)

        if val is not None:
             if val.ndim == 4:
                  axis = (0, 2, 3)
             else:
                  axis = 0
             
             s_min = val.min(axis=axis)
             s_max = val.max(axis=axis)
             s_mean = val.mean(axis=axis)
             s_std = val.std(axis=axis)
             
             stats[key] = {
                 "min": s_min.tolist(),
                 "max": s_max.tolist(),
                 "mean": s_mean.tolist(),
                 "std": s_std.tolist(),
                 "count": [int(val.shape[0])]
             }
             
    return stats
