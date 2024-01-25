import torch
from typing import Dict, List


def remove_layers(ckpt: Dict[str, torch.Tensor], removed_layers_idx: List[int]):
    if len(removed_layers_idx) <= 0:
        return ckpt

    new_ckpt = {}

    # check original layers_idx
    original_layers_idx = set()
    for k in ckpt.keys():
        if "blocks." in k:
            idx = int(k.split(".")[2])
            original_layers_idx.add(idx)
    original_layers_idx = sorted(list(original_layers_idx)) # [0,1,2,...,n]

    # check removed layers
    removed_layers_idx = sorted(removed_layers_idx)
    assert removed_layers_idx[-1] <= original_layers_idx[-1]

    # check remaining layers
    remain_layers_idx = []
    for idx in original_layers_idx:
        if idx not in removed_layers_idx:
            remain_layers_idx.append(idx)

    # build map
    rename_map = {}
    new_idx = 0
    for idx in remain_layers_idx:
        rename_map[f"blocks.{idx}."] = f"blocks.{new_idx}."
        new_idx += 1

    print("Original layers:", original_layers_idx)
    print("Remove layers:", removed_layers_idx)
    print("Remaining layers:", remain_layers_idx)

    # Stacking
    for k, v in ckpt.items():
        if 'pos_embed_t' in k or 'pos_embed_s' in k or 'mask_token' in k:
            continue
        elif 'blocks.' in k:
            if any(
                (s in k) for s in rename_map.keys()
            ):
                new_k = k
                for original_p, new_p in rename_map.items():
                    new_k = new_k.replace(original_p, new_p)
                new_ckpt[new_k] = v
                print(f"Rename \"{k}\" -> \"{new_k}\"")
        else:
            new_ckpt[k] = v
    
    return new_ckpt
