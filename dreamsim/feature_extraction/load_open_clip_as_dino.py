import torch
from .vision_transformer import vit_base, VisionTransformer
import open_clip
import os


def load_open_clip_as_dino(patch_size, load_dir="./models", l14=False):
    if l14:
        sd = torch.load(os.path.join(load_dir, 'open_clipl14_as_dino_vitl.pth.tar'), map_location='cpu')
        dino_vit = VisionTransformer(**sd['kwargs'])
        sd = sd['state_dict']
    else:
        dino_vit = vit_base(patch_size=patch_size)
        sd = torch.load(os.path.join(load_dir, f'open_clip_vitb{patch_size}_pretrain.pth.tar'))['state_dict']

    dino_vit.pos_drop = torch.nn.LayerNorm(dino_vit.embed_dim)
    proj = sd.pop('proj')
    dino_vit.load_state_dict(sd)
    for m in dino_vit.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.eps = 1e-5

    return dino_vit, proj


@torch.no_grad()
def _sanity_check(patch_size, l14=False):
    from PIL import Image

    if l14:
        dino_vit, proj = load_open_clip_as_dino(patch_size, l14=True)
        clip_all, _, preprocess = open_clip.create_model_and_transforms(f'ViT-L-{patch_size}',
                                                                        pretrained='laion400m_e31', cache_dir=".")
    else:
        dino_vit, proj = load_open_clip_as_dino(patch_size)
        clip_all, _, preprocess = open_clip.create_model_and_transforms(f'ViT-B-{patch_size}',
                                                                        pretrained='laion400m_e31', cache_dir=".")

    x = preprocess(Image.open('images/img_a_1.png'))[None, ...]

    # intermidiate representations
    cpre = []

    def clip_hook(module, x, y):
        if len(cpre) == 0:
            cpre.append(y)
        else:
            cpre.append(y.transpose(0, 1))  # batch and seq_len dimensions are flipped in CLIP

    clip_all.visual.ln_pre.register_forward_hook(clip_hook)  # checks
    [rb.register_forward_hook(clip_hook) for rb in clip_all.visual.transformer.resblocks]

    vpre = []

    def vit_hook(module, x, y):
        vpre.append(y)

    dino_vit.pos_drop.register_forward_hook(vit_hook)  # checks
    [b.register_forward_hook(vit_hook) for b in dino_vit.blocks]

    # forward
    ce = clip_all.encode_image(x)

    ve = dino_vit(x) @ proj

    for i, (c, v) in enumerate(zip(cpre, vpre)):
        delta = (c - v).abs()
        assert torch.isclose(c, v).all(), f'layer {i}:\ndelta={delta}\n{torch.isclose(c, v)}'
        print(f'{i}: {delta.abs().max()}')

    delta = (ce - ve).abs()
    assert torch.isclose(ce, ve).all(), f'final rep:\ndelta={delta}\n{torch.isclose(ce, ve)}'
    print(f'emb space: {delta.abs().max()}')


if __name__ == '__main__':
    _sanity_check(patch_size=32)
    #_sanity_check(patch_size=16)
    #_sanity_check(patch_size=14)
