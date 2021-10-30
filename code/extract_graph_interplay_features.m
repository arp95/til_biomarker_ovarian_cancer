% Original Author: Sara Arabyarmohammadi
% Modified by: Arpit Aggarwal

function [features_all_together, all_descriptions] = extract_graph_interplay_features(coords)
num_groups = length(coords);
features_all_together = [];
all_descriptions = [];

for i = 1:num_groups
    for j = i+1:num_groups
        for k = j+1:num_groups
            if (isempty(coords{i})==false && isempty(coords{j})==false && isempty(coords{k})==false)
                [features_all, description] = get_graph_interplay_features(coords{i}, coords{j}, coords{k});
                features_all_together = [features_all_together, features_all];
                all_descriptions = [all_descriptions, description];
            end
        end
    end
end