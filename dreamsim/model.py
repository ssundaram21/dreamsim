import json

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import os

from .constants import *
from .feature_extraction.extractor import ViTExtractor
import yaml
import peft
from peft import PeftModel, LoraConfig, get_peft_model
from .config import dreamsim_args, dreamsim_weights
import os
import zipfile
from packaging import version

peft_version = version.parse(peft.__version__)
min_version = version.parse("0.2.0")

if peft_version < min_version:
    raise RuntimeError(
        f"DreamSim requires peft version {min_version} or greater. "
        "Please update peft with 'pip install --upgrade peft'."
    )


class PerceptualModel(torch.nn.Module):
    def __init__(
        self,
        model_type: str = "dino_vitb16",
        feat_type: str = "cls",
        stride: str = "16",
        hidden_size: int = 1,
        lora: bool = False,
        baseline: bool = False,
        load_dir: str = "./models",
        normalize_embeds: bool = False,
        device: str = "cuda",
        **kwargs,
    ):
        """Initializes a perceptual model that returns the perceptual distance between two image tensors.
        Extracts features from one or more base ViT models and optionally passes them through an MLP.

        :param model_type: Comma-separated list of base ViT models. Accepted values are:
            ['dino_vits8', 'dino_vits16', 'dino_vitb8', 'dino_vitb16', 'clip_vitb16', 'clip_vitb32', 'clip_vitl14',
            'mae_vitb16', 'mae_vitl16', 'mae_vith14', 'open_clip_vitb16', 'open_clip_vitb32', 'open_clip_vitl14']
        :param feat_type: Which ViT feature to extract for each base model.
            Comma-separated list, same length as model_type. Accepted values are:
            'cls': The CLS token
            'embedding': The final layer tokens, extracted before the final projection head
            'last_layer':   The final layer tokens, extracted after the final projection head
            'cls_patch': The CLS token and global-pooled patch tokens, concatenated
        :param stride: Stride of first convolution layer for each base model (should match patch size).
            Comma-separated list, same length as model_type.
        :param lora: True means finetuning with LoRA. Replaces the MLP with an identity function.
        :param baseline: True means no finetuning; replaces the MLP with an identity function.
        :param hidden_size: Dimension of the final MLP hidden layer that extracted ViT features are passed to.
            If lora=True or baseline=True this argument is overriden and the MLP is replaced with an identity function.
        :param load_dir: Path to pretrained ViT checkpoints.
        :param normalize_embeds: If True, normalizes embeddings (i.e. divides by norm and subtracts mean).
        """
        super().__init__()
        self.model_list = model_type.split(",")
        self.feat_type_list = feat_type.split(",")
        self.stride_list = [int(x) for x in stride.split(",")]
        self.is_patch = "cls_patch" in self.feat_type_list
        self._validate_args()
        self.extract_feats_list = []
        self.extractor_list = nn.ModuleList()
        self.num_feats = []
        self.embed_size = 0
        self.hidden_size = hidden_size
        self.baseline = baseline
        for model_type, feat_type, stride in zip(
            self.model_list, self.feat_type_list, self.stride_list
        ):
            self.extractor_list.append(
                ViTExtractor(model_type, stride, load_dir, device=device)
            )
            extract_fn, num_feats = self._get_extract_fn(model_type, feat_type)
            self.extract_feats_list.append(extract_fn)
            self.num_feats.append(num_feats)
            self.embed_size += EMBED_DIMS[model_type][feat_type.split("_")[0]]
        self.lora = lora
        if self.lora or self.baseline:
            self.mlp = torch.nn.Identity()
        else:
            self.mlp = MLP(in_features=self.embed_size, hidden_size=self.hidden_size)
        self.normalize_embeds = normalize_embeds
        self.device = device

    def forward(self, img_a, img_b):
        """
        :param img_a: An RGB image passed as a (1, 3, 224, 224) tensor with values [0-1].
        :param img_b: Same as img_a.
        :return: A distance score for img_a and img_b. Higher means further/more different.
        """
        embed_a = self.embed(img_a)
        embed_b = self.embed(img_b)

        if self.feat_type_list[0] == "cls_patch":
            cls_a = embed_a[:, 0]
            patch_a = embed_a[:, 1:]
            cls_b = embed_b[:, 0]
            patch_b = embed_b[:, 1:]

            n = patch_a.shape[0]
            s = int(patch_a.shape[1] ** 0.5)
            patch_a_pooled = F.adaptive_avg_pool2d(
                patch_a.reshape(n, s, s, -1).permute(0, 3, 1, 2), (1, 1)
            ).squeeze()
            if len(patch_a_pooled.shape) == 1:
                patch_a_pooled = patch_a_pooled.unsqueeze(0)
            patch_b_pooled = F.adaptive_avg_pool2d(
                patch_b.reshape(n, s, s, -1).permute(0, 3, 1, 2), (1, 1)
            ).squeeze()
            if len(patch_b_pooled.shape) == 1:
                patch_b_pooled = patch_b_pooled.unsqueeze(0)

            embed_a = torch.cat((cls_a, patch_a_pooled), dim=-1)
            embed_b = torch.cat((cls_b, patch_b_pooled), dim=-1)
        return 1 - F.cosine_similarity(embed_a, embed_b, dim=-1)

    def embed(self, img):
        """
        Returns an embedding of img. The perceptual distance is the cosine distance between two embeddings. If the
        embeddings are normalized then L2 distance can also be used.
        """
        full_feats = (self.extract_feats_list[0](img, extractor_index=0)).squeeze()
        for i in range(1, len(self.extract_feats_list)):
            feats = (self.extract_feats_list[i](img, extractor_index=i)).squeeze()
            full_feats = torch.cat((full_feats, feats), dim=-1)
        embed = self.mlp(full_feats)

        if len(embed.shape) <= 1:
            embed = embed.unsqueeze(0)
        if len(embed.shape) <= 2 and self.is_patch:
            embed = embed.unsqueeze(0)

        if self.normalize_embeds:
            embed = (
                normalize_embedding_patch(embed)
                if self.is_patch
                else normalize_embedding(embed)
            )

        return embed

    def _validate_args(self):
        assert len(self.model_list) == len(self.feat_type_list) == len(self.stride_list)

        for model_type, feat_type, stride in zip(
            self.model_list, self.feat_type_list, self.stride_list
        ):
            if feat_type == "cls_patch" and "dino" not in model_type:
                raise ValueError(
                    f"cls_patch only available for dino_vitb16 and dinov2_vitb14, not {model_type}"
                )
            if feat_type == "embedding" and (
                "dino" in model_type or "mae" in model_type
            ):
                raise ValueError(f"{feat_type} not supported for {model_type}")
            if self.is_patch and feat_type != "cls_patch":
                # If cls_patch is specified for one model, it has to be specified for all.
                raise ValueError(
                    f"Cannot extract {feat_type} for {model_type}; cls_patch specified elsewhere."
                )

    def _get_extract_fn(self, model_type, feat_type):
        num_feats = 1
        if feat_type == "cls":
            extract_fn = self._extract_cls
        elif feat_type == "embedding":
            extract_fn = self._extract_embedding
        elif feat_type == "last_layer":
            extract_fn = self._extract_last_layer
        elif feat_type == "cls_patch":
            extract_fn = self._extract_cls_and_patch
            num_feats = 2
        else:
            raise ValueError(f"Feature type {feat_type} is not supported.")

        def extract(img, extractor_index):
            prep_img = self._preprocess(img, model_type)
            return extract_fn(prep_img, extractor_index=extractor_index)

        return extract, num_feats

    def _extract_cls_and_patch(self, img, extractor_index=0):
        layer = 11
        out = self.extractor_list[extractor_index].extract_descriptors(img, layer)
        return out

    def _extract_cls(self, img, extractor_index=0):
        layer = 11
        return self._extract_cls_and_patch(img, extractor_index)[:, :, :, 0, :]

    def _extract_embedding(self, img, extractor_index=0):
        return self.extractor_list[extractor_index].forward(img, is_proj=True)

    def _extract_last_layer(self, img, extractor_index=0):
        return self.extractor_list[extractor_index].forward(img, is_proj=False)

    def _preprocess(self, img, model_type):
        return transforms.Normalize(
            mean=self._get_mean(model_type), std=self._get_std(model_type)
        )(img)

    def _get_mean(self, model_type):
        if "dino" in model_type or "synclr" in model_type:
            return IMAGENET_DEFAULT_MEAN
        elif "open_clip" in model_type:
            return OPENAI_CLIP_MEAN
        elif "clip" in model_type:
            return OPENAI_CLIP_MEAN
        elif "mae" in model_type:
            return IMAGENET_DEFAULT_MEAN

    def _get_std(self, model_type):
        if "dino" in model_type or "synclr" in model_type:
            return IMAGENET_DEFAULT_STD
        elif "open_clip" in model_type:
            return OPENAI_CLIP_STD
        elif "clip" in model_type:
            return OPENAI_CLIP_STD
        elif "mae" in model_type:
            return IMAGENET_DEFAULT_STD


class MLP(torch.nn.Module):
    """
    MLP head with a single hidden layer and residual connection.
    """

    def __init__(self, in_features: int, hidden_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(in_features, self.hidden_size, bias=True)
        self.fc2 = torch.nn.Linear(self.hidden_size, in_features, bias=True)

    def forward(self, img):
        x = self.fc1(img)
        x = F.relu(x)
        return self.fc2(x) + img


def download_weights(cache_dir, dreamsim_type):
    """
    Downloads and unzips DreamSim weights.
    """
    dreamsim_required_ckpts = {
        "ensemble": [
            "dino_vitb16_pretrain.pth",
            "open_clip_vitb16_pretrain.pth.tar",
            "clip_vitb16_pretrain.pth.tar",
            "ensemble_lora",
        ],
        "dino_vitb16": ["dino_vitb16_pretrain.pth", "dino_vitb16_single_lora"],
        "dinov2_vitb14": ["dinov2_vitb14_pretrain.pth", "dinov2_vitb14_single_lora"],
        "open_clip_vitb32": [
            "open_clip_vitb32_pretrain.pth.tar",
            "open_clip_vitb32_single_lora",
        ],
        "clip_vitb32": ["clip_vitb32_pretrain.pth.tar", "clip_vitb32_single_lora"],
        "synclr_vitb16": ["synclr_vit_b_16.pth", "synclr_vitb16_single_lora"],
        "dino_vitb16_patch": ["dino_vitb16_pretrain.pth", "dino_vitb16_patch_lora"],
        "dinov2_vitb14_patch": [
            "dinov2_vitb14_pretrain.pth",
            "dinov2_vitb14_patch_lora",
        ],
    }

    def check(path):
        for required_ckpt in dreamsim_required_ckpts[dreamsim_type]:
            if not os.path.exists(os.path.join(path, required_ckpt)):
                return False
        return True

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if check(cache_dir):
        print(f"Using cached {cache_dir}")
    else:
        print("Downloading checkpoint")
        torch.hub.download_url_to_file(
            url=dreamsim_weights[dreamsim_type],
            dst=os.path.join(cache_dir, "pretrained.zip"),
        )
        print("Unzipping...")
        with zipfile.ZipFile(os.path.join(cache_dir, "pretrained.zip"), "r") as zip_ref:
            zip_ref.extractall(cache_dir)


def dreamsim(
    pretrained: bool = True,
    device="cuda",
    cache_dir="./models",
    normalize_embeds: bool = True,
    dreamsim_type: str = "ensemble",
    use_patch_model=False,
):
    """Initializes the DreamSim model. When first called, downloads/caches model weights for future use.

    :param pretrained: If True, downloads and loads DreamSim weights.
    :param cache_dir: Location for downloaded weights.
    :param device: Device for model.
    :param normalize_embeds: If True, normalizes embeddings (i.e. divides by norm and subtracts mean).
    :param dreamsim_type: The type of dreamsim model to use. The default is "ensemble" (default and best-performing)
                          which concatenates dino_vitb16, clip_vitb16, and open_clip_vitb16 embeddings. Other options
                          are "dino_vitb16", "clip_vitb32", "open_clip_vitb32", "dinov2_vitb14", and "synclr_vitb16",
                          which are finetuned single models.
    :param use_patch_model: If True, returns the model trained with CLS and patch features, not just CLS (only available for dino_vitb16 and dinov2_vitb14)
    :return:
        - PerceptualModel with DreamSim settings and weights.
        - Preprocessing function that converts a PIL image and to a (1, 3, 224, 224) tensor with values [0-1].
    """
    if use_patch_model:
        dreamsim_type += "_patch"
    # Get model settings and weights
    download_weights(cache_dir=cache_dir, dreamsim_type=dreamsim_type)

    # initialize PerceptualModel and load weights
    model_list = dreamsim_args["model_config"][dreamsim_type]["model_type"].split(",")
    ours_model = PerceptualModel(
        **dreamsim_args["model_config"][dreamsim_type],
        device=device,
        load_dir=cache_dir,
        normalize_embeds=normalize_embeds,
    )

    if dreamsim_type == "ensemble":
        tag = "ensemble_"
    elif use_patch_model:
        tag = f"{model_list[0]}_patch_"
    else:
        tag = f"{model_list[0]}_single_"

    with open(os.path.join(cache_dir, f"{tag}lora", "adapter_config.json"), "r") as f:
        adapter_config = json.load(f)
    lora_keys = ["r", "lora_alpha", "lora_dropout", "bias", "target_modules"]
    lora_config = LoraConfig(**{k: adapter_config[k] for k in lora_keys})
    ours_model = get_peft_model(ours_model, lora_config)

    if pretrained:
        load_dir = os.path.join(cache_dir, f"{tag}lora")
        ours_model = PeftModel.from_pretrained(
            ours_model.base_model.model, load_dir
        ).to(device)

    ours_model.eval().requires_grad_(False)

    # Define preprocessing function
    t = transforms.Compose(
        [
            transforms.Resize(
                (dreamsim_args["img_size"], dreamsim_args["img_size"]),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ]
    )

    def preprocess(pil_img):
        pil_img = pil_img.convert("RGB")
        return t(pil_img).unsqueeze(0)

    return ours_model, preprocess


def normalize_embedding(embed):
    embed = (embed.T - torch.mean(embed, dim=1)).T
    return (embed.T / torch.norm(embed, dim=1)).T


def normalize_embedding_patch(embed):
    mean_matrix = torch.mean(embed, dim=2).unsqueeze(1)
    embed = (embed.mT - mean_matrix).mT
    normed_matrix = torch.norm(embed, dim=2).unsqueeze(1)
    return (embed.mT / normed_matrix).mT


EMBED_DIMS = {
    "dino_vits8": {"cls": 384, "last_layer": 384},
    "dino_vits16": {"cls": 384, "last_layer": 384},
    "dino_vitb8": {"cls": 768, "last_layer": 768},
    "dino_vitb16": {"cls": 768, "last_layer": 768},
    "dinov2_vitb14": {"cls": 768, "last_layer": 768},
    "clip_vitb16": {"cls": 768, "embedding": 512, "last_layer": 768},
    "clip_vitb32": {"cls": 768, "embedding": 512, "last_layer": 512},
    "clip_vitl14": {"cls": 1024, "embedding": 768, "last_layer": 768},
    "mae_vitb16": {"cls": 768, "last_layer": 768},
    "mae_vitl16": {"cls": 1024, "last_layer": 1024},
    "mae_vith14": {"cls": 1280, "last_layer": 1280},
    "open_clip_vitb16": {"cls": 768, "embedding": 512, "last_layer": 768},
    "open_clip_vitb32": {"cls": 768, "embedding": 512, "last_layer": 768},
    "open_clip_vitl14": {"cls": 1024, "embedding": 768, "last_layer": 768},
    "synclr_vitb16": {"cls": 768, "last_layer": 768},
}
