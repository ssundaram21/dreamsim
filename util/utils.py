import torch
from torchvision import transforms


def get_preprocess(model_type):
    if 'lpips' in model_type:
        return 'LPIPS'
    elif 'dists' in model_type:
        return 'DISTS'
    elif 'psnr' in model_type:
        return 'PSNR'
    elif 'ssim' in model_type:
        return 'SSIM'
    elif 'clip' in model_type or 'open_clip' in model_type or 'dino' in model_type or 'mae' in model_type:
        return 'DEFAULT'
    else:
        return 'DEFAULT'


def get_preprocess_fn(preprocess, load_size, interpolation):
    if preprocess == "LPIPS":
        t = transforms.ToTensor()
        return lambda pil_img: t(pil_img.convert("RGB")) / 0.5 - 1.
    else:
        if preprocess == "DEFAULT":
            t = transforms.Compose([
                transforms.Resize((load_size, load_size), interpolation=interpolation),
                transforms.ToTensor()
            ])
        elif preprocess == "DISTS":
            t = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
        elif preprocess == "SSIM" or preprocess == "PSNR":
            t = transforms.ToTensor()
        else:
            raise ValueError("Unknown preprocessing method")
        return lambda pil_img: t(pil_img.convert("RGB"))



