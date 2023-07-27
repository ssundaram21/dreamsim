import torch
import torch.nn.functional as F
from torchvision import transforms
import os
from .feature_extraction.extractor import ViTExtractor
import yaml
from peft import PeftModel, LoraConfig, get_peft_model
from .feature_extraction.vit_wrapper import ViTConfig, ViTModel
from .config import dreamsim_args, dreamsim_weights
import os
import zipfile


class PerceptualModel(torch.nn.Module):
    def __init__(self, model_type: str = "dino_vitb16", feat_type: str = "cls", stride: str = '16', hidden_size: int = 1,
                 lora: bool = False, baseline: bool = False, load_dir: str = "./models", normalize_embeds: bool = False,
                 device: str = "cuda", **kwargs):
        """ Initializes a perceptual model that returns the perceptual distance between two image tensors.
        Extracts features from one or more base ViT models and optionally passes them through an MLP.

        :param model_type: Comma-separated list of base ViT models. Accepted values are:
            ['dino_vits8', 'dino_vits16', 'dino_vitb8', 'dino_vitb16', 'clip_vitb16', 'clip_vitb32', 'clip_vitl14',
            'mae_vitb16', 'mae_vitl16', 'mae_vith14', 'open_clip_vitb16', 'open_clip_vitb32', 'open_clip_vitl14']
        :param feat_type: Which ViT feature to extract for each base model.
            Comma-separated list, same length as model_type. Accepted values are:
            'cls': The CLS token
            'embedding': The final layer tokens, extracted before the final projection head
            'last_layer':   The final layer tokens, extracted after the final projection head
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
        self.model_list = model_type.split(',')
        self.feat_type_list = feat_type.split(',')
        self.stride_list = [int(x) for x in stride.split(',')]
        self._validate_args()
        self.extract_feats_list = []
        self.extractor_list = []
        self.embed_size = 0
        self.hidden_size = hidden_size
        self.baseline = baseline
        for model_type, feat_type, stride in zip(self.model_list, self.feat_type_list, self.stride_list):
            self.extractor_list.append(
                ViTExtractor(model_type, stride, load_dir, device=device)
            )
            self.extract_feats_list.append(
                self._get_extract_fn(model_type, feat_type)
            )
            self.embed_size += EMBED_DIMS[model_type][feat_type]
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
        if self.normalize_embeds:
            embed = normalize_embedding(embed)
        return embed

    def _validate_args(self):
        assert len(self.model_list) == len(self.feat_type_list) == len(self.stride_list)
        for model_type, feat_type, stride in zip(self.model_list, self.feat_type_list, self.stride_list):
            if feat_type == "embedding" and ("dino" in model_type or "mae" in model_type):
                raise ValueError(f"{feat_type} not supported for {model_type}")

    def _get_extract_fn(self, model_type, feat_type):
        if feat_type == "cls":
            extract_fn = self._extract_cls
        elif feat_type == "embedding":
            extract_fn = self._extract_embedding
        elif feat_type == "last_layer":
            extract_fn = self._extract_last_layer
        else:
            raise ValueError(f"Feature type {feat_type} is not supported.")

        def extract(img, extractor_index):
            prep_img = self._preprocess(img, model_type)
            return extract_fn(prep_img, extractor_index=extractor_index)

        return extract

    def _extract_cls(self, img, extractor_index=0):
        layer = 11
        return self.extractor_list[extractor_index].extract_descriptors(img, layer)[:, :, :, 0, :]

    def _extract_embedding(self, img, extractor_index=0):
        return self.extractor_list[extractor_index].forward(img, is_proj=True)

    def _extract_last_layer(self, img, extractor_index=0):
        return self.extractor_list[extractor_index].forward(img, is_proj=False)

    def _preprocess(self, img, model_type):
        return transforms.Normalize(mean=self._get_mean(model_type), std=self._get_std(model_type))(img)

    def _get_mean(self, model_type):
        if "dino" in model_type:
            return (0.485, 0.456, 0.406)
        elif "open_clip" in model_type:
            return (0.48145466, 0.4578275, 0.40821073)
        elif "clip" in model_type:
            return (0.48145466, 0.4578275, 0.40821073)
        elif "mae" in model_type:
            return (0.485, 0.456, 0.406)

    def _get_std(self, model_type):
        if "dino" in model_type:
            return (0.229, 0.224, 0.225)
        elif "open_clip" in model_type:
            return (0.26862954, 0.26130258, 0.27577711)
        elif "clip" in model_type:
            return (0.26862954, 0.26130258, 0.27577711)
        elif "mae" in model_type:
            return (0.229, 0.224, 0.225)


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
        "ensemble": ["dino_vitb16_pretrain.pth", "dino_vitb16_lora",
                     "open_clip_vitb16_pretrain.pth.tar", "open_clip_vitb16_lora",
                     "clip_vitb16_pretrain.pth.tar", "clip_vitb16_lora"],
        "dino_vitb16": ["dino_vitb16_pretrain.pth", "dino_vitb16_single_lora"],
        "open_clip_vitb32": ["open_clip_vitb32_pretrain.pth.tar", "open_clip_vitb32_single_lora"],
        "clip_vitb32": ["clip_vitb32_pretrain.pth.tar", "clip_vitb32_single_lora"]
    }

    def check(path):
        for required_ckpt in dreamsim_required_ckpts[dreamsim_type]:
            if not os.path.exists(os.path.join(path, required_ckpt)):
                return False
        return True

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if check(cache_dir):
        print(f"Using cached {cache_dir}")
    else:
        print("Downloading checkpoint")
        torch.hub.download_url_to_file(url=dreamsim_weights[dreamsim_type],
                                       dst=os.path.join(cache_dir, "pretrained.zip"))
        print("Unzipping...")
        with zipfile.ZipFile(os.path.join(cache_dir, "pretrained.zip"), 'r') as zip_ref:
            zip_ref.extractall(cache_dir)


def dreamsim(pretrained: bool = True, device="cuda", cache_dir="./models", normalize_embeds: bool = True,
             dreamsim_type: str = "ensemble"):
    """ Initializes the DreamSim model. When first called, downloads/caches model weights for future use.

    :param pretrained: If True, downloads and loads DreamSim weights.
    :param cache_dir: Location for downloaded weights.
    :param device: Device for model.
    :param normalize_embeds: If True, normalizes embeddings (i.e. divides by norm and subtracts mean).
    :param dreamsim_type: The type of dreamsim model to use. The default is "ensemble" (default and best-performing)
                          which concatenates dino_vitb16, clip_vitb16, and open_clip_vitb16 embeddings. Other options
                          are "dino_vitb16", "clip_vitb32", and "open_clip_vitb32" which are finetuned single models.
    :return:
        - PerceptualModel with DreamSim settings and weights.
        - Preprocessing function that converts a PIL image and to a (1, 3, 224, 224) tensor with values [0-1].
    """
    # Get model settings and weights
    download_weights(cache_dir=cache_dir, dreamsim_type=dreamsim_type)

    # initialize PerceptualModel and load weights
    model_list = dreamsim_args['model_config'][dreamsim_type]['model_type'].split(",")
    ours_model = PerceptualModel(**dreamsim_args['model_config'][dreamsim_type], device=device, load_dir=cache_dir,
                                 normalize_embeds=normalize_embeds)
    for extractor in ours_model.extractor_list:
        lora_config = LoraConfig(**dreamsim_args['lora_config'])
        model = get_peft_model(ViTModel(extractor.model, ViTConfig()), lora_config)
        extractor.model = model

    tag = "" if dreamsim_type == "ensemble" else "single_"
    if pretrained:
        for extractor, name in zip(ours_model.extractor_list, model_list):
            load_dir = os.path.join(cache_dir, f"{name}_{tag}lora")
            extractor.model = PeftModel.from_pretrained(extractor.model, load_dir).to(device)
            extractor.model.eval().requires_grad_(False)

    ours_model.eval().requires_grad_(False)

    # Define preprocessing function
    t = transforms.Compose([
        transforms.Resize((dreamsim_args['img_size'], dreamsim_args['img_size']),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    def preprocess(pil_img):
        pil_img = pil_img.convert('RGB')
        return t(pil_img).unsqueeze(0)

    return ours_model, preprocess


def normalize_embedding(embed):
    if len(embed.shape) <= 1:
        embed = embed.unsqueeze(0)
    embed = (embed.T / torch.norm(embed, dim=1)).T
    return (embed.T - torch.mean(embed, dim=1)).T

EMBED_DIMS = {
    'dino_vits8': {'cls': 384, 'last_layer': 384},
    'dino_vits16': {'cls': 384, 'last_layer': 384},
    'dino_vitb8': {'cls': 768, 'last_layer': 768},
    'dino_vitb16': {'cls': 768, 'last_layer': 768},
    'clip_vitb16': {'cls': 768, 'embedding': 512, 'last_layer': 768},
    'clip_vitb32': {'cls': 768, 'embedding': 512, 'last_layer': 512},
    'clip_vitl14': {'cls': 1024, 'embedding': 768, 'last_layer': 768},
    'mae_vitb16': {'cls': 768, 'last_layer': 768},
    'mae_vitl16': {'cls': 1024, 'last_layer': 1024},
    'mae_vith14': {'cls': 1280, 'last_layer': 1280},
    'open_clip_vitb16': {'cls': 768, 'embedding': 512, 'last_layer': 768},
    'open_clip_vitb32': {'cls': 768, 'embedding': 512, 'last_layer': 768},
    'open_clip_vitl14': {'cls': 1024, 'embedding': 768, 'last_layer': 768}
}

