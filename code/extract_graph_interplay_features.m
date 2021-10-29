% Original Author: Sara

function [features_All_together,all_descriptions] = extract_graph_interplay_features(coords)
numGroups = length(coords);
features_All_together = [];
all_descriptions = [];

for i = 1:numGroups
    for j = i+1:numGroups
        for k = j+1:numGroups
            if (isempty(coords{i})== false && isempty(coords{j})== false && isempty(coords{k})== false)
                [features_All,description] = getGraphInterplayFeatures_v5(coords{i},coords{j},coords{k});
                features_All_together=[features_All_together,features_All];
                all_descriptions=[all_descriptions,description];
            end
        end
    end
end