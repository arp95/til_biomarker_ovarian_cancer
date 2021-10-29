% Original Author: Sepideh Azarianpour

function SA_drawGraphsAndConvexHull_all(I, V30, V41, coords, colors, r, a, visFile)
num_groups = length(coords);
MM = cell(num_groups, 1);
for i=1:num_groups
    alpha = a(i);
    [~, ~, ~, ~, group_matrix] = construct_node(struct('centroid_r', coords{i}(:,2)', 'centroid_c', coords{i}(:,1)'), alpha, r);
    MM{i} = group_matrix;
end

figure
imshow((V41), 'Border', 'tight');
hold on;
drawGraph_standard(coords, MM, colors);
drawGraph_boundary_standard(coords, colors, a, r, 3, 3);
saveas(gcf, [visFile, '_4.png'])

figure
imshow(ones(size(V41)), 'Border', 'tight');
hold on;
drawGraph_standard(coords, MM, colors);
drawGraph_convexHull_standard(coords, colors, a, r, 3, 3);
saveas(gcf, [visFile, '_5.png'])

figure
imshow(ones(size(V41)), 'Border', 'tight');
hold on;
drawGraph_standard(coords, MM, colors);
drawGraph_boundary_standard(coords, colors, a, r, 3, 3);
saveas(gcf, [visFile, '_6.png'])