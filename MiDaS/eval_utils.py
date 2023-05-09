# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from utils import read_pfm


def pfm_to_pil(path):
    data, _ = read_pfm(path)
    data = Image.fromarray(data)
    return data


def load_image(img_path, mode):
    """
    Loads the input or the GT into a PIL.Image object.
    Input images can be stored in uint8 or uin16 format.
    GT images are encoded in grayscale images, so we decode them.
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if mode == "depth":
        mask = img == 0
        f_img = (img - 1).astype(np.float32) / 128.0
        f_img[mask] = 0.0
        return Image.fromarray(f_img)
    else:
        if img.dtype == np.uint16:
            tonemap_fct_16_bit = lambda x : cv2.sqrt(x.astype(np.float32)).astype(np.uint8)
            img = tonemap_fct_16_bit(img)
        return Image.fromarray(img).convert(mode)


def visualize_depth_map(image, target, prediction, output_filename):
    image = np.array(image)
    target = np.array(target)
    prediction = np.array(prediction)
    cmap = copy.deepcopy(plt.cm.jet)
    cmap.set_bad(color="black")
    _min, _max = np.nanmin(target), np.nanmax(target)
    
    fig, ax = plt.subplots(1, 3, figsize=(50, 50))
    ax[0].imshow(image)
    ax[1].imshow(target, cmap=cmap, vmin = _min, vmax = _max)
    ax[2].imshow(prediction, cmap=cmap, vmin = _min, vmax = _max)

    fig.tight_layout()
    fig.savefig(output_filename)
    plt.close()