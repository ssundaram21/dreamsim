import torch
from .vision_transformer import vit_base, VisionTransformer
import os


def load_synclr_as_dino(patch_size, load_dir="./models", l14=False):
    sd = torch.load(os.path.join(load_dir, f'synclr_vit_b_{patch_size}.pth'))['model']
    dino_vit = vit_base(patch_size=patch_size)
    new_sd = dict()

    for k, v in sd.items():
        new_key = k[14:]  # strip "module.visual" from key
        new_sd[new_key] = v

    dino_vit.load_state_dict(new_sd)
    return dino_vit
