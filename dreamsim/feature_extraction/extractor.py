import types
from typing import List, Tuple
import math
import torch
import torch.nn.modules.utils as nn_utils
from torch import nn
import os
from .load_clip_as_dino import load_clip_as_dino
from .load_open_clip_as_dino import load_open_clip_as_dino
from .vision_transformer import DINOHead
from .load_mae_as_vit import load_mae_as_vit

"""
Mostly copy-paste from https://github.com/ShirAmir/dino-vit-features.
"""


class ViTExtractor:
    """ This class facilitates extraction of features, descriptors, and saliency maps from a ViT.

    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(self, model_type: str = 'dino_vits8', stride: int = 4, load_dir: str = "./models",
                 device: str = 'cuda'):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224 | mae_vitb16 | mae_vitl16 |
                          mae_vith14]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param load_dir: location of pretrained ViT checkpoints.
        """
        self.model_type = model_type
        self.device = device
        self.model = ViTExtractor.create_model(model_type, load_dir)
        if type(self.model) is tuple:
            self.proj = self.model[1]
            self.model = self.model[0]
        self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride).eval().to(self.device)
        self.p = self.model.patch_embed.patch_size
        if type(self.p) is tuple:
            self.p = self.p[0]
        self.stride = self.model.patch_embed.proj.stride
        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str, load_dir: str = "./models") -> nn.Module:
        """
        :param model_type: a string specifying which model to load. ['dino_vits8' | 'dino_vits16' | 'dino_vitb8' |
            'dino_vitb16' | 'clip_vitb16' | 'clip_vitb32' | 'clip_vitl14' | 'mae_vitb16' | 'mae_vitl16' | 'mae_vith14' |
            'open_clip_vitb16' | 'open_clip_vitb32' | 'open_clip_vitl14']
        :param load_dir: location of pretrained ViT checkpoints.
        :return: the model
        """
        if 'dino' in model_type:
            torch.hub.set_dir(load_dir)
            model = torch.hub.load('facebookresearch/dino:main', model_type)
            if model_type == 'dino_vitb16':
                sd = torch.load(os.path.join(load_dir, 'dino_vitb16_pretrain.pth'), map_location='cpu')
                proj = DINOHead(768, 2048)
                proj.mlp[0].weight.data = sd['student']['module.head.mlp.0.weight']
                proj.mlp[0].bias.data = sd['student']['module.head.mlp.0.bias']
                proj.mlp[2].weight.data = sd['student']['module.head.mlp.2.weight']
                proj.mlp[2].bias.data = sd['student']['module.head.mlp.2.bias']
                proj.mlp[4].weight.data = sd['student']['module.head.mlp.4.weight']
                proj.mlp[4].bias.data = sd['student']['module.head.mlp.4.bias']
                proj.last_layer.weight.data = sd['student']['module.head.last_layer.weight']
                model = (model, proj)
        elif 'open_clip' in model_type:
            if model_type == 'open_clip_vitb16':
                model = load_open_clip_as_dino(16, load_dir)
            elif model_type == 'open_clip_vitb32':
                model = load_open_clip_as_dino(32, load_dir)
            elif model_type == 'open_clip_vitl14':
                model = load_open_clip_as_dino(14, load_dir, l14=True)
            else:
                raise ValueError(f"Model {model_type} not supported")
        elif 'clip' in model_type:
            if model_type == 'clip_vitb16':
                model = load_clip_as_dino(16, load_dir)
            elif model_type == 'clip_vitb32':
                model = load_clip_as_dino(32, load_dir)
            elif model_type == 'clip_vitl14':
                model = load_clip_as_dino(14, load_dir, l14=True)
            else:
                raise ValueError(f"Model {model_type} not supported")
        elif 'mae' in model_type:
            model = load_mae_as_vit(model_type, load_dir)
        else:
            raise ValueError(f"Model {model_type} not supported")
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """
        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        if stride == patch_size or (stride,stride) == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in
                    stride]), f'stride {stride} should divide patch_size {patch_size}'

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(ViTExtractor._fix_pos_enc(patch_size, stride), model)
        return model

    def forward(self, x, is_proj=False):
        if is_proj:
            if 'clip' in self.model_type:
                return self.model(x) @ self.proj.to(self.device)
            elif 'dino' in self.model_type:
                if self.model_type == 'dino_vitb16':
                    self.proj = self.proj.to(self.device)
                    return self.proj(self.model(x))
                raise NotImplementedError
            elif 'mae' in self.model_type:
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            return self.model(x)

    def _get_drop_hook(self, drop_rate):

        def dt_pre_hook(module, tkns):
            bp_ = torch.ones_like(tkns[0][0, :, 0])
            bp_[1:] = drop_rate  # do not drop the [cls] token!
            tkns = tkns[0][:, torch.bernoulli(bp_) > 0.5, :]
            return tkns

        return dt_pre_hook

    def fix_random_seeds(suffix):
        """
        Fix random seeds.
        """
        seed = hash(suffix) % (2 ** 31 - 1)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    def _get_hook(self):
        """
        generate a hook method for a specific block and facet.
        """
        def _hook(model, input, output):
            self._feats.append(output)
        return _hook

    def _register_hooks(self, layers: List[int],drop_rate=0) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        """
        if drop_rate > 0:
            self.hook_handlers.append(self.model.blocks[0].register_forward_pre_hook(self._get_drop_hook(drop_rate)))
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                self.hook_handlers.append(block.register_forward_hook(self._get_hook()))

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, drop_rate=0) -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :return : tensor of features with shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, drop_rate)
        try:
            _ = self.model(batch)
            self._unregister_hooks()
        except Exception as e:
            self._unregister_hooks()
            raise e
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats

    def extract_descriptors(self, batch: torch.Tensor, layer: int = 11, drop_rate=0) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :return: tensor of descriptors. Bxlx1xtxd' where d' is the dimension of the descriptors.
        """
        if type(layer) is not list:
            layer = [ layer ]

        self._extract_features(batch, layer, drop_rate)
        x = torch.stack(self._feats, dim=1)
        x = x.unsqueeze(dim=2) #Bxlx1xtxd # Default to facet = "token", always include CLS token
        desc = x.permute(0, 1, 3, 4, 2).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=2)  # Bxlx1xtx(dxh)
        return desc
