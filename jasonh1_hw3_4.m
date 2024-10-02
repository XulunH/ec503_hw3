
% EC 503 - HW 3
% DP-Means implementation on synthetic Gaussian data from Problem 3.2(a)

clear, clc, close all;
rng('default'); % For reproducibility

%% Generate Gaussian data (Problem 3.2(a)):
% Means
mu1 = [2, 2];
mu2 = [-2, 2];
mu3 = [0, -3.25];

% Covariance matrices
sigma1 = 0.02 * eye(2);
sigma2 = 0.05 * eye(2);
sigma3 = 0.07 * eye(2);

% Number of points per cluster
points_per_cluster = 50;

% Generate data for each cluster
cluster1 = mvnrnd(mu1, sigma1, points_per_cluster);
cluster2 = mvnrnd(mu2, sigma2, points_per_cluster);
cluster3 = mvnrnd(mu3, sigma3, points_per_cluster);

% Combine data into one dataset
DATA = [cluster1; cluster2; cluster3];
num_points = size(DATA, 1);

% True labels for visualization
true_labels = [ones(points_per_cluster,1); 2*ones(points_per_cluster,1); 3*ones(points_per_cluster,1)];

% Scatter plot of the generated data with true labels
figure;
scatter(cluster1(:,1), cluster1(:,2), 'r', 'filled'); hold on;
scatter(cluster2(:,1), cluster2(:,2), 'g', 'filled');
scatter(cluster3(:,1), cluster3(:,2), 'b', 'filled');
title('Scatter Plot of Generated Gaussian Data (True Clusters)');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Cluster 1','Cluster 2','Cluster 3');
hold off;

%% DP Means method for different lambda values
lambda_values = [0.15, 0.4, 3, 20];

for idx = 1:length(lambda_values)
    LAMBDA = lambda_values(idx);
    fprintf('\nRunning DP-means with lambda = %.2f\n', LAMBDA);

    % Initialization
    convergence_threshold = 1e-4;
    total_indices = 1:num_points;

    % Cluster count
    K = 1;

    % Class indicators/labels
    Z = ones(num_points, 1);

    % Means MU: initial mean is the mean of all data
    MU = mean(DATA, 1);

    converged = 0;
    t = 0;
    while ~converged
        t = t + 1;
        % Store old cluster information for convergence check
        K_old = K;
        MU_old = MU;

        %% Per Data Point:
        for i = 1:num_points

            %% CODE 1 - Calculate distance from current point to all currently existing clusters
            differences = MU - DATA(i, :); % Differences between point and cluster means
            distances = sqrt(sum(differences.^2, 2)); % Euclidean distances to cluster centers

            %% CODE 2 - Find the distance to closest cluster. If the distance is more than LAMBDA start a new cluster else update Z(i)
            [dist_min, cluster_idx] = min(distances);
            if dist_min > LAMBDA
                % Start a new cluster
                K = K + 1;
                Z(i) = K;
                MU = [MU; DATA(i, :)]; % Add new cluster mean
            else
                % Assign to closest cluster
                Z(i) = cluster_idx;
            end
        end

        %% CODE 3 - Form new sets of points (clusters)
        % One cell of L contains indices of points belonging to a single cluster
        L = arrayfun(@(c) find(Z == c), 1:K, 'UniformOutput', false);

        %% CODE 4 - Recompute means per cluster
        MU = zeros(K, size(DATA, 2));
        for c = 1:K
            if ~isempty(L{c})
                MU(c, :) = mean(DATA(L{c}, :), 1);
            else
                MU(c, :) = NaN; % Mark empty clusters
            end
        end

        % Remove empty clusters
        empty_clusters = any(isnan(MU), 2);
        if any(empty_clusters)
            MU = MU(~empty_clusters, :);
            L = L(~empty_clusters);
            K = K - sum(empty_clusters);
            % Adjust Z to reflect new cluster indices
            old_to_new_cluster_idx = cumsum(~empty_clusters);
            Z = old_to_new_cluster_idx(Z);
        end

        %% CODE 5 - Test for convergence: number of clusters doesn't change and means stay the same
        delta_MU = max(sqrt(sum((MU - MU_old).^2, 2)));
        if delta_MU < convergence_threshold && K == K_old
            converged = 1;
        end
    end

    %% CODE 6 - Plot final clusters after convergence 
    % Plot the clusters
    figure;
    hold on;
    colors = lines(K);
    for c = 1:K
        scatter(DATA(L{c}, 1), DATA(L{c}, 2), 36, colors(c, :), 'filled');
        % Plot cluster centers
        plot(MU(c, 1), MU(c, 2), 'kx', 'MarkerSize', 12, 'LineWidth', 2);
    end
    title(sprintf('DP-means Clustering Result (\\lambda = %.2f)', LAMBDA));
    xlabel('Feature 1');
    ylabel('Feature 2');
    legend_entries = arrayfun(@(x) sprintf('Cluster %d', x), 1:K, 'UniformOutput', false);
    legend([legend_entries, 'Cluster Centers']);
    hold off;

    fprintf('Converged after %d iterations with %d clusters.\n', t, K);
end
