# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved. Except portions as noted which are Copyright (c) 2023 OpenGVLab and licensed under the MIT license found in LICENSE.
import numpy as np
import torch
import math
import random

from PIL import Image, ImageDraw
from torchvision import transforms as T
from torchvision.transforms import Compose, RandAugment, RandomResizedCrop, Resize, ToPILImage


# Imagenet's mean and std.
pixel_mean = [123.675, 116.28, 103.53]
pixel_std = [58.395, 57.12, 57.375]

# Reshape for broadcasting.
pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)


def convert_to_rgb(image):
    return image.convert("RGB")

def _transform_train_aug():
    return Compose([
        ToPILImage(),
        Resize(scale=random.random() / 2 + 0.5),
        convert_to_rgb,
        RandAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
    ])

def _transform_test():
    return Compose([
        ToPILImage(),
        convert_to_rgb,
    ])


def standardize_image(img):
    """Standardize image pixel values."""
    return (torch.Tensor(np.array(img)).permute(2, 0, 1) - pixel_mean) / pixel_std


def get_visual_transform(
        img, 
        factor: int = 28, 
        min_pixels: int = 56 * 56, 
        max_pixels: int = 14 * 14 * 4 * 1280, 
        augment=False
    ):
    img = np.array(img)

    if augment:
        visual_transform = _transform_train_aug()
    else:
        visual_transform = _transform_test()

    img = visual_transform(img)
    w, h = img.size
    h_bar, w_bar = smart_resize(h, w, factor, min_pixels, max_pixels)
    img = img.resize((w_bar, h_bar))

    # Standardize pixel values.
    img = standardize_image(img)
    imgs = [img]
    return imgs

# copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar