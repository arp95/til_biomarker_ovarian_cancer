"""
Clean the epi/stroma masks by applying morphological operations.
"""


# header files needed
import cv2
import numpy as np
import glob


# parameters
masks = glob.glob("results/epithelium_stroma_masks/*")
print(masks)

# main code
for mask in masks:
    image = cv2.imread(mask, 0)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]
    image_inv = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # filter using contour area and remove small noise
    cnts = cv2.findContours(image_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 5500:
            cv2.drawContours(image_inv, [c], -1, (0, 0, 0), -1)

    # filter using contour area and remove small noise
    output_mask = 255 - image_inv
    cnts = cv2.findContours(output_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 5500:
            cv2.drawContours(output_mask, [c], -1, (0, 0, 0), -1)

    # write mask
    cv2.imwrite(mask, output_mask)