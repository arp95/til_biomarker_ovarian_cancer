%% Author: Arpit Aggarwal
clc
close all
clear 
addpath(genpath('pwd'))

%% hard-coded paths
%patches_dir = "../../ovarian_cancer_results/patches_sample/";
%patches = dir(fullfile(patches_dir, '*.png'));
%epi_stroma_masks_dir = "../../ovarian_cancer_results/epi_stroma_masks_final/";
%nuclei_masks_dir = "../../ovarian_cancer_results/nuclei_masks_final/";
%histoqc_masks_dir = "";
%results_images_dir = "../../ovarian_cancer_results/til_masks_sample/";
%results_features_dir = "./";
%til_model_path = "../../ovarian_cancer_model_files/lymp_svm_matlab_wsi.mat";
%draw_option = 0;

%% hard-coded paths on HPC
patches_dir = "/mnt/rstor/CSE_BME_AXM788/home/axa1399/tcga_ovarian_cancer/patches/patches/";
patches = dir(fullfile(patches_dir, '*.png'));
epi_stroma_masks_dir = "/mnt/rstor/CSE_BME_AXM788/home/axa1399/tcga_ovarian_cancer/epi_stroma_masks/";
nuclei_masks_dir = "/mnt/rstor/CSE_BME_AXM788/home/axa1399/tcga_ovarian_cancer/nuclei_masks/";
histoqc_masks_dir = "/mnt/rstor/CSE_BME_AXM788/home/axa1399/tcga_ovarian_cancer/histoqc_masks/";
results_images_dir = "/mnt/rstor/CSE_BME_AXM788/home/axa1399/tcga_ovarian_cancer/til_masks/";
results_features_dir = "/mnt/rstor/CSE_BME_AXM788/home/axa1399/tcga_ovarian_cancer/results_new/features/";
til_model_path = "/mnt/rstor/CSE_BME_AXM788/home/axa1399/til_biomarker_ovarian_cancer/model_files/lymp_svm_matlab_wsi.mat";
draw_option = 0;


%% get til features
for index = 32001:42000
    filename = patches(index).name;
    epi_stroma_mask_path = epi_stroma_masks_dir + filename;
    nuclei_mask_path = nuclei_masks_dir + filename;
    histoqc_mask_path = histoqc_masks_dir + filename;
    results_image_path = char(results_images_dir + filename);

    if isfile(nuclei_mask_path)
        %% read patches, epi/stroma mask, nuclei mask, HistoQC mask, TIL model
        image = im2double(imread(patches_dir + filename));
        epi_stroma_mask = im2double(imread(epi_stroma_mask_path));
        nuclei_mask = im2double(imread(nuclei_mask_path));
        histoqc_mask = im2double(imread(histoqc_mask_path));
        epi_mask = (epi_stroma_mask.*histoqc_mask);
        stroma_mask = ((1-epi_stroma_mask).*histoqc_mask);
        nuclei_mask = (nuclei_mask.*histoqc_mask);
        til_model = load(til_model_path);

        %% run til pipeline
        [features] = extract_til_features(image, nuclei_mask, histoqc_mask, epi_mask, stroma_mask, til_model, draw_option, results_image_path);
        %filename = extractBefore(filename, ".png");
        %writematrix(features, results_features_dir + filename + '.csv');
    end
end