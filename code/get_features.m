function [features, feature_names] = get_features(coords, alpha, r, neigborhood_size, grouping_threshold)

if nargin<3
    r=.185;
end
if nargin<4
    neigborhood_size = 5;
end
if nargin<5
    grouping_threshold = .005;
end

%% Variable initialization
% This value defines the maximum number of close neighbors used to compute
% the intersection between clusters. It was set to 5 for performance.
maxNeigborsInters=5;
num_groups = length(coords);
feature_names = cell(1, num_groups);
clust_per_group = zeros(1, num_groups);

%% Building clusters
groups = struct;
for i=1:num_groups
    groups(i).nodes = coords{i};
    [~, ~, ~, ~, group_matrix] = construct_node(struct('centroid_r',coords{i}(:, 2)', 'centroid_c', coords{i}(:, 1)'), alpha(i), r);
    [~, ~, groups(i).clusters] = network_components(group_matrix);
        
    % Discarding clusters with less than 3 nodes
    groups(i).clusters(cellfun(@length, groups(i).clusters) <= 2) = [];
        
    % Getting cluster properties (centroids, polygons, areas, and densities)
    [groups(i).clusterCentroids, groups(i).clusterPolygons,...
    groups(i).areas, groups(i).densities] = get_cluster_properties(groups(i).clusters, groups(i).nodes);
    
     % Getting clusters by group
     clust_per_group(i) = length(groups(i).clusters);
     feature_names{i} = ['NumClusters_G' num2str(i)];
end

%% Identifying closest neighbors per group
for i=1:num_groups
     groups(i).neighborsPerGroup = get_closest_neighbors_by_group(groups, i);
end

%% Identifying absolute closest neighbors
max_num_clust = max(clust_per_group);
for i=1:num_groups
    groups(i).absoluteClosest = get_absolute_closest_neighbors(groups, i, max_num_clust);
end

%% Extracting cluster-related features
[densFeat,densFeatNames] = getClusterDensityMeasures_v2(groups);
[intersClustFeat, intersClustFeatNames] = getClusterIntersectionMeasures(groups,maxNeigborsInters);
[richFeat, richFeatNames] = getNeighborhoodMeasures(groups,neigborhoodSize,maxNumClust);

%% Extracting group-related features
[graphFeat, graphFeatNames] = getGraphFeatures(groups);
[intersGroupFeat, intersGroupFeatNames] = getIntersectionGroups(groups);
[groupingFeat, groupingFeatNames] = getGroupingFactorByGroup(groups, groupingThreshold);

%% Compiling features
features = [clustPerGroup, densFeat, intersClustFeat, richFeat, graphFeat, intersGroupFeat, groupingFeat];
feature_names = [feature_names, densFeatNames, intersClustFeatNames, richFeatNames, graphFeatNames, intersGroupFeatNames, groupingFeatNames];
end