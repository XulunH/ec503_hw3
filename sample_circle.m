% Parameters for sample_circle
num_clusters = 3;
points_per_cluster = 500 * ones(num_clusters, 1);

% Generate concentric ring data
[data, label] = sample_circle(num_clusters, points_per_cluster);

% Scatter plot of concentric ring dataset
figure;
scatter(data(:, 1), data(:, 2), 10, label, 'filled');
title('Concentric Ring Clusters');
xlabel('X');
ylabel('Y');
grid on;
axis equal;

% Parameters for k-means clustering
k = 3;
num_trials = 10;
threshold = 1e-4;

% Store WCSS for each trial
wcss_trials = zeros(num_trials, 1);
all_cluster_assignments = cell(num_trials, 1);
all_means = cell(num_trials, 1);

% Run k-means clustering with multiple random initializations
for trial = 1:num_trials
    % Random initialization of means
    initial_means = min(data) + (max(data) - min(data)) .* rand(k, size(data, 2));
    
    % Perform k-means clustering
    [cluster_assignments, means] = kmeans_clustering(data, k, initial_means, threshold);
    
    % Compute WCSS for this clustering
    wcss_trials(trial) = compute_wcss(data, cluster_assignments, means);
    
    % Store results
    all_cluster_assignments{trial} = cluster_assignments;
    all_means{trial} = means;
end

% Find the best trial with the minimum WCSS
[~, best_trial] = min(wcss_trials);
best_cluster_assignments = all_cluster_assignments{best_trial};
best_means = all_means{best_trial};

% Plot best clustering result
figure;
colors = lines(k); % Get distinct colors for clusters
hold on;
for j = 1:k
    scatter(data(best_cluster_assignments == j, 1), data(best_cluster_assignments == j, 2), 10, colors(j, :), 'filled');
end
scatter(best_means(:, 1), best_means(:, 2), 100, 'kx', 'LineWidth', 2); % Plot cluster centers
title('K-means Clustering of Concentric Rings');
xlabel('X');
ylabel('Y');
legend(arrayfun(@(x) ['Cluster ' num2str(x)], 1:k, 'UniformOutput', false), 'Location', 'Best');
grid on;
axis equal;
hold off;

function [data ,label] = sample_circle( num_cluster, points_per_cluster )
% Function to sample 2-D circle-shaped clusters
% Input:
% num_cluster: the number of clusters 
% points_per_cluster: a vector of [num_cluster] numbers, each specify the
% number of points in each cluster 
% Output:
% data: sampled data points. Each row is a data point;
% label: ground truth label for each data points.
%
% EC 503: Learning from Data
% Instructor: Prakash Ishwar
% HW 3, Problem 3.2(f) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin == 0
  num_cluster = 2;
  points_per_cluster = 500*ones(num_cluster,1);
end
if nargin == 1
   points_per_cluster = 500*ones(num_cluster,1);
end
points_per_cluster=points_per_cluster(:);

data = zeros([sum(points_per_cluster), 2]);
label = zeros(sum(points_per_cluster),1);
idx = 1;
bandwidth = 0.1;

for k = 1 : num_cluster
    theta = 2 * pi * rand(points_per_cluster(k), 1);
    rho = k + randn(points_per_cluster(k), 1) * bandwidth;
    [x, y] = pol2cart(theta, rho);
    data(idx:idx+points_per_cluster(k)-1,:) = [x, y];
    label(idx:idx+points_per_cluster(k)-1)=k;
    idx = idx + points_per_cluster(k);
end
