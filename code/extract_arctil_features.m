function [patch,features]=extract_arctil_features(image, nuclei_mask, histoqc_mask, epi_mask, stroma_mask, E_HM, til_model, draw_option, results_file)

%% patch is blank if majority are empty pixels, epithelial, stroma or invasive tumour front
epi_mask_value = sum(sum(im2double(epi_mask)))/(size(epi_mask, 1)*size(epi_mask, 2));
stroma_mask_value = sum(sum(im2double(stroma_mask)))/(size(stroma_mask, 1)*size(stroma_mask, 2));
epi_plus_stroma = epi_mask_value + stroma_mask_value
nuclei_mask_value = sum(sum(im2double(nuclei_mask)))/(size(nuclei_mask, 1)*size(nuclei_mask, 2));
if epi_plus_stroma<0.2
    patch = 'Blank';
else
    if (eps+epi_mask_value)/(eps+stroma_mask_value)>3
        patch = 'Epithelial';
        ROI = epi_mask;
    end
    %% stromal patches
    if (eps+stroma_mask_value)/(eps+epi_mask_value)>3
        patch = 'Stromal';
        ROI = stroma_mask;
    end
    %% tumor edge patches
    if (eps+epi_mask_value)/(eps+stroma_mask_value)<3 && (eps+stroma_mask_value)/(eps+epi_mask_value)<3
        patch = 'TumourEdge';
        ROI = stroma_mask+epi_mask;
    end
end

patch
if strcmp(patch, 'Blank')
    fprintf('Empty Patch\n')
    features = zeros(1172, 1);
else
    %% classify the cells in ROI
    draw_option = 1
    [nuclei_centroids, nuclei_features, ~] = get_nuclei_features(image, ROI.*nuclei_mask);
    is_lymphocyte = (predict(til_model.model, nuclei_features(:,1:7))) == 1;
    nuclei_centroids_rounded = round(nuclei_centroids);
    num_nuclei_centroids = length(nuclei_centroids_rounded);
    epi_nuclei = false(num_nuclei_centroids, 1);
    features = []
    size(nuclei_centroids_rounded)
    size(is_lymphocyte)

    % if less nuclei then empty patch, otherwise find features
    if num_nuclei_centroids < 3
        fprintf('Empty patch \n')
        features = zeros(1172, 1);
    else
        for c=1:num_nuclei_centroids
            epi_nuclei(c) = epi_mask(nuclei_centroids_rounded(c, 2), nuclei_centroids_rounded(c, 1));
        end

        % get epi-TILs, epi non-TILs, stroma TILs and stroma non-TILs
        coords = {   nuclei_centroids_rounded(~is_lymphocyte & epi_nuclei,:),...
            nuclei_centroids_rounded(is_lymphocyte & ~epi_nuclei,:),...
            nuclei_centroids_rounded(is_lymphocyte & epi_nuclei,:),...
            nuclei_centroids_rounded(~is_lymphocyte & ~epi_nuclei,:),...
            };

        % nuclei graphs
        alpha=0.37*ones(1,4);
        r=.185;
        
        %% draw centroids, graphs, convex hull for all families
        if draw_option == 1
            classes = zeros(1, num_nuclei_centroids);
            classes(is_lymphocyte & ~epi_nuclei) = 1;
            classes(is_lymphocyte & epi_nuclei) = 2;
            classes(~is_lymphocyte & ~epi_nuclei) = 3;
            colors = {...
                [1, 0.54, 0],...
                [0 .81 .91],...
                [0.45, 0.8, 0],...
                [.1 0 .7]};
            V30 = ESW_maker2(epi_mask, stroma_mask, histoqc_mask);
            V40 = (V30 + image)/2;
            V41 = ROImaker(V40, ROI);
            drawNucContoursByClass_SA2(nuclei_mask, V41, nuclei_centroids, classes, colors);
            saveas(gcf, [results_file, '_3.png'])
            SA_drawGraphsAndConvexHull_all(image, V30, V41, coords, colors, r, alpha, results_file)
        end
         %% extracting features
        %[features, feature_names] = get_features(coords, alpha, r);
        %%featNames = reshape(featNames,[size(featNames,2), size(featNames,1)]);
    end    
end