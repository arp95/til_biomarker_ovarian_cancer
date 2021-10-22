clc
close all
clear 
addpath(genpath('pwd'))
%% read patches, epi/stroma mask, nuclei mask and HistoQC mask
image = im2double(imread('../ArcTIL/Patch Examples/case_3.png'));
nuclei_mask = im2double(imread('../ArcTIL/Patch Examples/nuclei_case_3.png'));
epi_stroma_mask = im2double(imread('../ArcTIL/Patch Examples/epi_case_3.png'))
histoqc_mask = (ones(3000, 3000));
epi_mask = (epi_stroma_mask.*histoqc_mask);
stroma_mask = ((1-epi_stroma_mask).*histoqc_mask);

%% load svm model for classifying TILs vs non-TILs
hh = '../biomarkers_ovarian_cancer/model_files/lymp_svm_matlab_wsi.mat';
til_model = load(hh);

results_file = '../biomarkers_ovarian_cancer/results/';
[patch, features] = extract_arctil_features(image, nuclei_mask, histoqc_mask, epi_mask, stroma_mask, epi_mask, til_model, 1, results_file);