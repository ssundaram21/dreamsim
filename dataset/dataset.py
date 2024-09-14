from torch.utils.data import Dataset
from util.utils import get_preprocess_fn
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os


class TwoAFCDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", load_size: int = 224,
                 interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
                 preprocess: str = "DEFAULT", **kwargs):
        self.root_dir = root_dir
        self.csv = pd.read_csv(os.path.join(self.root_dir, "data.csv"))
        self.csv = self.csv[self.csv['votes'] >= 6] # Filter out triplets with less than 6 unanimous votes
        self.split = split
        self.load_size = load_size
        self.interpolation = interpolation
        self.preprocess_fn = get_preprocess_fn(preprocess, self.load_size, self.interpolation)
        
        if self.split == "train" or self.split == "val" or self.split == "test":
            self.csv = self.csv[self.csv["split"] == split]
        elif split == 'test_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == True]
        elif split == 'test_no_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == False]
        else:
            raise ValueError(f'Invalid split: {split}')

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        id = self.csv.iloc[idx, 0]
        p = self.csv.iloc[idx, 2].astype(np.float32)
        img_ref = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 4])))
        img_left = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 5])))
        img_right = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 6])))
        return img_ref, img_left, img_right, p, id

