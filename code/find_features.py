'''
Author: Arpit Aggarwal
Python script to find the features for each file
'''


# header files
import os
import glob
import csv
import numpy as np


# parameters to update
files_path = "./files.csv"
results_features_path = "/mnt/rstor/CSE_BME_AXM788/home/axa1399/tcga_ovarian_cancer/results/features/"
results_final_features_path = "/mnt/rstor/CSE_BME_AXM788/home/axa1399/tcga_ovarian_cancer/results/final_features/"

# find the images for finding their corresponding features
flag = -1
files = []
with open(files_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        if flag == -1:
            array = row
            files.append(array[0])
        else:
            array = row
            files.append(array[0])
print("Files to find features for: " + str(len(files)))


results_features_files = glob.glob(results_features_path + "*")
for file in files:
    features = []
    for results_features_file in results_features_files:
        if file[:-4] in results_features_file:
            current_features = []
            current_features_float = []
            sum = 0
            flag = -1
            with open(results_features_file, newline='') as csvfile:
                spamreader = csv.reader(csvfile)
                for row in spamreader:
                    if flag == -1:
                        current_features = row
                        for index in range(0, len(current_features)):
                            sum = sum + float(current_features[index])
                            current_features_float.append(float(current_features[index]))
            if sum > 1:
                features.append(current_features_float)
    # features for the corresponding file
    print(file + " " + str(len(features)))
    features_for_file = []
    if len(features) > 0:
        for index1 in range(0, 892):
            value = 0.0
            for index2 in range(0, len(features)):
                value += float(features[index2][index1])
            value = float(value) / float(len(features))
            features_for_file.append(value)
    else:
        for index in range(0, 892):
            features_for_file.append(0)

    # write csv for the file
    with open(results_final_features_path + file[:-4] + ".csv", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(features_for_file)