from PIL import Image
from dreamsim import dreamsim
from torchvision import transforms
import torch
import os

img_size = 224
t = transforms.Compose([
    transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor()
])


def preprocess(img):
    img = img.convert('RGB')
    return t(img).unsqueeze(0)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, _ = dreamsim(pretrained=True, device=device)

# Load images
img_ref = preprocess(Image.open('images/ref_1.png')).to(device)
img_0 = preprocess(Image.open('images/img_a_1.png')).to(device)
img_1 = preprocess(Image.open('images/img_b_1.png')).to(device)

# Get distance
d0 = model(img_ref, img_0)
d1 = model(img_ref, img_1)

print(d0, d1)

# # Get embeddings
# embed_ref = model.embed(img_ref)
# embed_0 = model.embed(img_0)
# embed_1 = model.embed(img_1)


