%% Author: Arpit Aggarwal
clc
close all
clear 
addpath(genpath('pwd'))

%% hard-coded paths
image_path = "../../ovarian_cancer_results/patches_example/TCGA-25-2401_75000_51000.png";
epi_stroma_mask_path = "../../ovarian_cancer_results/epithelium_stroma_masks_example/TCGA-25-2401_75000_51000.png";
nuclei_mask_path = "../../ovarian_cancer_results/nuclei_masks/TCGA-25-2401_75000_51000.png";
histoqc_mask_path = "";
til_model_path = "../../ovarian_cancer_model_files/lymp_svm_matlab_wsi.mat";
results_file = char("../../ovarian_cancer_results/TCGA-25-2401_75000_51000");
draw_option = 0

%% read patches, epi/stroma mask, nuclei mask, HistoQC mask, TIL model
image = im2double(imread(image_path));
epi_stroma_mask = im2double(imread(epi_stroma_mask_path))
nuclei_mask = im2double(imread(nuclei_mask_path));
histoqc_mask = (ones(3000, 3000));
epi_mask = (epi_stroma_mask.*histoqc_mask);
stroma_mask = ((1-epi_stroma_mask).*histoqc_mask);
til_model = load(til_model_path);

%% run til pipeline
[patch, features] = extract_til_features(image, nuclei_mask, histoqc_mask, epi_mask, stroma_mask, til_model, draw_option, results_file);