"""
Original Author: Arpit Aggarwal
Description of the file: Script for extracting histoqc masks after running the main histoqc pipeline.
"""

# header files
import numpy as np
import glob
import cv2
print("Header files loaded...")


# get the options selected by user
patches_dir = "/scratch/users/sxa786/uh_endometrium_cancer/patches/"
histoqc_masks_dir = "/mnt/rstor/CSE_BME_AXM788/home/axa1399/histoqc_mask_output_uh_endometrial/"
results_dir = "/scratch/users/sxa786/uh_endometrium_cancer/histoqc_masks/"
patches = glob.glob(patches_dir + "*")
patches = patches[40000:50000]


# extract histoqc masks
for patch in patches:
    image_path = patch
    image = cv2.imread(image_path)
    
    image_name = image_path.split("/")[-1]
    image_split = image_name[:-4]
    image_split = image_split.split("_")
    full_image_name = image_split[0]
    index1 = int(image_split[len(image_split)-2])
    index2 = int(image_split[len(image_split)-1])

    print(histoqc_masks_dir + full_image_name + ".tif_mask_use.png")
    mask = cv2.imread(histoqc_masks_dir + full_image_name + ".tif_mask_use.png")
    index1 = int(index1 / 32.0)
    index2 = int(index2 / 32.0)
    mask = mask[index2:index2+94, index1:index1+94]
    mask = cv2.resize(mask, (3000, 3000))
    cv2.imwrite(results_dir + image_name, mask)
print("Done!")