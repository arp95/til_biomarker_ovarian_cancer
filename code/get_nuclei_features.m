% Original Author: Sepideh Azarianpour

function [centroids, features, feature_names, bounding_boxes] = get_nuclei_features(image, mask)
mask=mask(:, :, 1);
mask = logical(mask);
mask = bwareaopen(mask, 30);
image = normalize_staining(image);
gray_image = rgb2gray(image);
region_properties = regionprops(mask, gray_image,'Centroid','Area',...
    'BoundingBox','Eccentricity','EquivDiameter','Image',...
    'MajorAxisLength','MaxIntensity','MeanIntensity','MinIntensity',...
    'MinorAxisLength','Orientation','PixelValues');
centroids = cat(1, region_properties.Centroid);
med_red = [];
entropy_red = [];
bounding_boxes = [];

% get features for each nuclei
nuclei_num = size(region_properties, 1);
for i=1:nuclei_num
    nucleus = region_properties(i);
    bbox = nucleus.BoundingBox;
    bbox = [round(bbox(1)) round(bbox(2)) (bbox(3) - 1) (bbox(4) - 1)];
    bounding_boxes = [bounding_boxes; bbox];
    roi = image(bbox(2) : bbox(2) + bbox(4), bbox(1) : bbox(1) + bbox(3), :);
    per = bwperim(nucleus.Image);
    
    gray = rgb2gray(roi);
    R = roi(:,:,1);    
    R = R(nucleus.Image == 1);

    grayPix = gray(nucleus.Image == 1);
    perPix = gray(per==1);
    
    % Intensity features:
    med_red = [med_red; median(double(R))];

    % Entropies
    entropy_red=[entropy_red; get_nuclei_entropy(R)];
end
% ratio of axes
ratio_axes=[region_properties.MajorAxisLength]./[region_properties.MinorAxisLength];

% get nuclei features and feature names
features = horzcat([region_properties.Area]',[region_properties.Eccentricity]', ratio_axes', med_red, entropy_red, double([region_properties.MinIntensity]'), double([region_properties.MaxIntensity]'));
feature_names=[{'Area','Eccentricity','RatioAxes','MedianRed','EntropyRed', 'MinIntensity', 'MaxIntensity'}];
end