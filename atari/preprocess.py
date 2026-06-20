import numpy as np
import torch
from stable_pretraining import data as dt


def _to_chw(obs):
    # (H, W, C) -> (C, H, W)
    arr = np.asarray(obs)
    arr = arr.transpose(2, 0 ,1)
    return torch.as_tensor(arr, dtype=torch.uint8)

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)