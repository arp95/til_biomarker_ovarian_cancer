"""
Using Cheng's mode for epi/stroma segmentation. Updated script for my use case.
"""


# header files needed
import openslide
from unet import *
from glob import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import sys
import os
from matplotlib import cm
from torch.utils.data import DataLoader


# parameters
model_path = "code/misc/epi_seg_unet.pth"
savedir = "code"

# load model
#device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
net = torch.load(model_path, map_location=device)
net.eval()


# function to generate epi/stroma masks on patch basis
def get_epithelium_stroma_mask(im_path, patch_dim, level_magnification=1):

    # read image
    ts = openslide.OpenSlide(im_path)
    size0, size1, size2, size3 = ts.level_dimensions
    size = ts.level_dimensions[level_magnification]
    im_name = im_path.split('/')[-1]
    output_mask_name = f'{savedir}/{im_name}'
    output_mask_name_1 = output_mask_name.replace(f'.svs', f'_epistroma_mask.png')

    # read patch and get its binary mask output
    for index2 in range(0, int(size[1]), patch_dim):
        for index1 in range(0, int(size[0]), patch_dim):
            index1 = 9000
            index2 = 15000
            patch = ts.read_region((index1*4, index2*4), level_magnification, (patch_dim, patch_dim)).convert('RGB')
            patch_level0 = ts.read_region((index1*4, index2*4), 0, (patch_dim*4, patch_dim*4)).convert('RGB')
            output_patch = get_patch_epithelium_stroma_mask(patch)

            # save patch mask
            if index1 == 9000 and index2 == 15000:
                path = output_mask_name_1.replace(".png", "_" + str(index2) + "_" + str(index1) + ".png")
                save_patch_epithelium_stroma_mask(output_patch, path)
                path = output_mask_name.replace(f'.svs', "_" + str(index2) + "_" + str(index1) + ".png")
                save_patch(patch_level0, path)


# function to return epi/stroma mask for a given patch
def get_patch_epithelium_stroma_mask(patch):
    # get original patch dimensions
    np_original_patch = np.array(patch).astype(np.uint8)
    h = int(np_original_patch.shape[0])
    w = int(np_original_patch.shape[1])

    # resize patch and get output mask
    np_patch = np.array(patch).astype(np.uint8)
    output_patch_mask = np.zeros((h, w)).astype(np.uint8)
    
    for index1 in range(0, h, 231):
        for index2 in range(0, w, 231):
            np_patch_part = np_patch[index1:index1+256, index2:index2+256]
            h_part = int(np_patch_part.shape[0])
            w_part = int(np_patch_part.shape[1])

            np_patch_part = np_patch_part.transpose((2, 0, 1))
            np_patch_part = np_patch_part / 255
            tensor_patch = torch.from_numpy(np_patch_part)
            x = tensor_patch.unsqueeze(0)
            x = x.to(device, dtype=torch.float32)
            output = net(x)
            output = torch.sigmoid(output)
            pred = output.detach().squeeze().cpu().numpy()
            mask_pred = (pred>.5).astype(np.uint8)
            pil_mask_pred = Image.fromarray(mask_pred*255)
            np_mask_pred = (np.array(pil_mask_pred)/255).astype(np.uint8)

            # update output
            output_patch_mask[index1:index1+h_part, index2:index2+w_part] = np_mask_pred
    return output_patch_mask


# function to save epi/stroma mask for a given patch
def save_patch_epithelium_stroma_mask(patch, output_path):
    h = patch.shape[0]
    w = patch.shape[1]
    patch = Image.fromarray(patch*255).resize((w*4, h*4), Image.BICUBIC)
    patch.save(output_path)

# function to save patch
def save_patch(patch, output_path):
    Image.fromarray(np.array(patch).astype(np.uint8)).save(output_path)


# run code
if __name__ == '__main__':
    im_paths = glob("data/*.svs")
    for im_path in im_paths:
        get_epithelium_stroma_mask(im_path, patch_dim=250, level_magnification=1)
    print("Done!")