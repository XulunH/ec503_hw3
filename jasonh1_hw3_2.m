% Parameters for synthetic data generation
mu1 = [2; 2];
mu2 = [-2; 2];
mu3 = [0; -3.25];

Sigma1 = 0.02 * eye(2);
Sigma2 = 0.05 * eye(2);
Sigma3 = 0.07 * eye(2);

numPoints = 50; % Number of points per cluster

% Generate data for each cluster
rng('default'); % For reproducibility
cluster1 = mvnrnd(mu1, Sigma1, numPoints);
cluster2 = mvnrnd(mu2, Sigma2, numPoints);
cluster3 = mvnrnd(mu3, Sigma3, numPoints);

% Combine all clusters
data = [cluster1; cluster2; cluster3];

% Scatter plot of generated Gaussian data
figure;
scatter(cluster1(:,1), cluster1(:,2), 'r'); hold on;
scatter(cluster2(:,1), cluster2(:,2), 'g');
scatter(cluster3(:,1), cluster3(:,2), 'b');
title('Synthetic Gaussian Clusters');
xlabel('X1');
ylabel('X2');
legend('Cluster 1', 'Cluster 2', 'Cluster 3');
hold off;

k = 3;
threshold = 1e-4;

initial_means_1 = [3, 3; -4, -1; 2, -4];
initial_means_2 = [-0.14,2.61;3.15,-0.84;-3.28,-1.58];

[cluster_assignments_1, means_1] = kmeans_clustering(data, k, initial_means_1, threshold);
plot_clusters(data,cluster_assignments_1,means_1,k);
[cluster_assignments_2, means_2] = kmeans_clustering(data, k, initial_means_2, threshold);
plot_clusters(data,cluster_assignments_2,means_2,k);

num_trials = 10;

% Store WCSS for each trial
wcss_values = zeros(num_trials, 1);
all_cluster_assignments = cell(num_trials, 1);
all_means = cell(num_trials, 1);

% Run k-means for multiple initializations
for trial = 1:num_trials
    % Random initialization of means
    initial_means = min(data) + (max(data) - min(data)) .* rand(k, size(data, 2));
    
    % Perform k-means clustering
    [cluster_assignments, means] = kmeans_clustering(data, k, initial_means, threshold);
    
    % Compute WCSS for this clustering
    wcss_values(trial) = compute_wcss(data, cluster_assignments, means);
    
    % Store results
    all_cluster_assignments{trial} = cluster_assignments;
    all_means{trial} = means;
end

% Find the best trial with the minimum WCSS
[~, best_trial] = min(wcss_values);
best_cluster_assignments = all_cluster_assignments{best_trial};
best_means = all_means{best_trial};

% Report WCSS values for each trial
disp('WCSS values for each trial:');
disp(wcss_values');

% Report the best WCSS value
disp(['Best WCSS value: ', num2str(wcss_values(best_trial))]);

% Plot best clustering
plot_clusters(data, best_cluster_assignments, best_means, k);

% Range of k values to test
Krange = 2:10;
num_trials = 10;
threshold = 1e-4;

% Store WCSS for each k
best_wcss = zeros(length(Krange), 1);

% Loop over each k value in Krange
for idx = 1:length(Krange)
    k = Krange(idx);
    
    % Store WCSS for each trial
    wcss_trials = zeros(num_trials, 1);
    
    % Run k-means for multiple initializations
    for trial = 1:num_trials
        % Random initialization of means
        initial_means = min(data) + (max(data) - min(data)) .* rand(k, size(data, 2));
        
        % Perform k-means clustering
        [cluster_assignments, means] = kmeans_clustering(data, k, initial_means, threshold);
        
        % Compute WCSS for this clustering
        wcss_trials(trial) = compute_wcss(data, cluster_assignments, means);
    end
    
    % Store the best WCSS for this k
    best_wcss(idx) = min(wcss_trials);
end

% Plot WCSS values against k
figure;
plot(Krange, best_wcss, '-o', 'LineWidth', 2);
title('WCSS vs. Number of Clusters (k)');
xlabel('Number of Clusters (k)');
ylabel('WCSS');
grid on;

% Read NBA data from the uploaded file
filename = 'NBA_stats_2018_2019.xlsx';
nba_data = readtable(filename, 'VariableNamingRule', 'preserve');
% Extract Points Per Game (PPG) and Minutes Per Game (MPG) columns
PPG = nba_data.PPG; % Assume the column name is 'PPG'
MPG = nba_data.MPG; % Assume the column name is 'MPG'

% Form a 2D dataset from PPG and MPG
data = [MPG, PPG];

% Scatter plot of PPG vs. MPG
figure;
scatter(MPG, PPG, 'filled');
title('PPG vs. MPG for NBA Players (2018-2019)');
xlabel('Minutes Per Game (MPG)');
ylabel('Points Per Game (PPG)');
grid on;

% Parameters for k-means clustering
k = 10;
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
    scatter(data(best_cluster_assignments == j, 1), data(best_cluster_assignments == j, 2), [], colors(j, :), 'filled');
end
scatter(best_means(:, 1), best_means(:, 2), 100, 'kx', 'LineWidth', 2); % Plot cluster centers
title('K-means Clustering of NBA Players (PPG vs. MPG)');
xlabel('Minutes Per Game (MPG)');
ylabel('Points Per Game (PPG)');
legend(arrayfun(@(x) ['Cluster ' num2str(x)], 1:k, 'UniformOutput', false), 'Location', 'Best'); % Removed 'Cluster Centers'
grid on;
hold off;



function [cluster_assignments, means] = kmeans_clustering(data, k, initial_means, threshold)

    % Initialize cluster means
    means = initial_means;

    % Variable to keep track of cluster assignments
    cluster_assignments = zeros(size(data, 1), 1);

    % K-means clustering algorithm
    converged = false;
    while ~converged
        % Step 1: Assign points to the nearest cluster center
        for i = 1:size(data, 1)
            distances = sum((data(i, :) - means) .^ 2, 2);
            [~, cluster_assignments(i)] = min(distances);
        end

        % Step 2: Update cluster means
        new_means = zeros(k, size(data, 2));
        for j = 1:k
            cluster_points = data(cluster_assignments == j, :);
            if ~isempty(cluster_points)
                new_means(j, :) = mean(cluster_points, 1);
            else
                new_means(j, :) = means(j, :); % Avoid empty cluster case
            end
        end

        % Check for convergence
        if max(abs(new_means - means), [], 'all') < threshold
            converged = true;
        end

        means = new_means; % Update means
    end
end

function wcss = compute_wcss(data, cluster_assignments, means)
    wcss = 0;
    for j = 1:max(cluster_assignments)
        cluster_points = data(cluster_assignments == j, :);
        wcss = wcss + sum(sum((cluster_points - means(j, :)) .^ 2));
    end
end

% Function to plot clusters
function plot_clusters(data, cluster_assignments, means, k)
    figure;
    colors = {'r', 'g', 'b'};
    hold on;
    for j = 1:k
        scatter(data(cluster_assignments == j, 1), data(cluster_assignments == j, 2), colors{j});
    end
    scatter(means(:, 1), means(:, 2), 100, 'kx', 'LineWidth', 2);
    title('Best K-means Clustering Result');
    xlabel('X1');
    ylabel('X2');
    legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster Centers');
    hold off;
end
