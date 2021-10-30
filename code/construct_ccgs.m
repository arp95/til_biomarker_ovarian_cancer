% Original Author: Sara Arabyarmohammadi

function [VX, VY, x, y, edges, params] = construct_ccgs(bounds, alpha, r)
if nargin < 2,
    alpha = 0.5;
end
if nargin < 3
    r = 0.2;
end

% distance and probability matrices
X = [bounds(:,1),bounds(:, 2)];
D = pdist(X, 'euclidean');
P = D.^-alpha;

%define edges
edges = triu(true(length(bounds)), 1) & squareform(r < P);

% get edge locations
[xx, yy] = find(edges);
VX = [bounds(xx,1)'; bounds(yy,1)'];
VY = [bounds(xx,2)'; bounds(yy,2)'];

% get node locations
idx = unique([xx, yy],'rows', 'first');
x = [bounds(idx, 1)]';
y = [bounds(idx, 2)]';
params.r = r;
params.alpha = alpha;