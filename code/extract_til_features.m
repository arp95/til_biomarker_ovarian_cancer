% Author: Arpit Aggarwal

function [features]=extract_til_features(image, nuclei_mask, histoqc_mask, epi_mask, stroma_mask, til_model, draw_option, results_file)
%% get til features
[nuclei_centroids, nuclei_features, ~] = get_nuclei_features(image, nuclei_mask);
is_lymphocyte = (predict(til_model.model, nuclei_features(:,1:7))) == 1;
nuclei_centroids_rounded = round(nuclei_centroids);
epi_nuclei = false(length(nuclei_centroids_rounded), 1);
features = []

% if less nuclei then empty patch, otherwise find features
if length(nuclei_centroids_rounded) < 40
    fprintf('Empty patch \n');
    features = zeros(892, 1);
else
    for c=1:length(nuclei_centroids_rounded)
        epi_nuclei(c) = epi_mask(nuclei_centroids_rounded(c, 2), nuclei_centroids_rounded(c, 1));
    end

    size(nuclei_centroids_rounded)
    size(epi_nuclei)

    % get epi-TILs, epi non-TILs, stroma TILs and stroma non-TILs
    coords = {nuclei_centroids_rounded(~is_lymphocyte & epi_nuclei,:), nuclei_centroids_rounded(is_lymphocyte & ~epi_nuclei,:), nuclei_centroids_rounded(is_lymphocyte & epi_nuclei,:), nuclei_centroids_rounded(~is_lymphocyte & ~epi_nuclei,:),};
    length(nuclei_centroids_rounded(~is_lymphocyte & epi_nuclei,:))
    length(nuclei_centroids_rounded(is_lymphocyte & ~epi_nuclei,:))
    length(nuclei_centroids_rounded(is_lymphocyte & epi_nuclei,:))
    length(nuclei_centroids_rounded(~is_lymphocyte & ~epi_nuclei,:))
    if (length(nuclei_centroids_rounded(~is_lymphocyte & epi_nuclei,:)) < 10) || (length(nuclei_centroids_rounded(is_lymphocyte & ~epi_nuclei,:)) < 10) || (length(nuclei_centroids_rounded(is_lymphocyte & epi_nuclei,:)) < 10) || (length(nuclei_centroids_rounded(~is_lymphocyte & ~epi_nuclei,:)) < 10)
        % not useful patch for feature extraction
        fprintf("Not enough nuclei for extracting graph interplay features!");
        features = zeros(892, 1);
    else
        % extract graph interplay features
        [features_all_together, all_descriptions] = extract_graph_interplay_features(coords)
        features = features_all_together
        length(features_all_together)
    end

    %% draw centroids, graphs, convex hull for all families
    if draw_option == 1
        classes = zeros(1, length(nuclei_centroids_rounded));
        classes(is_lymphocyte & ~epi_nuclei) = 1;
        classes(is_lymphocyte & epi_nuclei) = 2;
        classes(~is_lymphocyte & ~epi_nuclei) = 3;
        colors = {[0, 0, 0], [0 .81 .91], [0 .81 .91], [0, 0, 0]};
        V30 = ESW_maker2(epi_mask, stroma_mask, histoqc_mask);
        V40 = (V30 + image)/2;
        V41 = ROImaker(V40, epi_mask+stroma_mask);
        drawNucContoursByClass_SA2(nuclei_mask, V41, nuclei_centroids, classes, colors);
        saveas(gcf, [results_file, '.png'])
    end
end