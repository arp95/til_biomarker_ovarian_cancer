"""
Original Author: Arpit Aggarwal
Description of the file: Script for extracting til masks after running the main til extraction pipeline.
"""

# header files
import numpy as np
import glob
import cv2
print("Header files loaded...")


# get the options selected by user
til_dir = "/mnt/rstor/CSE_BME_AXM788/home/axa1399/tcga_ovarian_cancer/til_masks/"
results_dir = "/mnt/rstor/CSE_BME_AXM788/home/axa1399/tcga_ovarian_cancer/til_masks_new_25/"
patches = glob.glob(til_dir + "*")
patches = patches[10000:15000]


# extract til masks
for patch in patches:
    image_path = patch
    image = cv2.imread(image_path)
    image_name = image_path.split("/")[-1]
    
    for i in range(0, 25):
        dilated = cv2.dilate(image.copy(), None, iterations=i + 1)
    cv2.imwrite(results_dir + image_name, dilated)
print("Done!")