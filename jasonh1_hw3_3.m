% Range of k values to test (Krange from Problem 3.2(d))
Krange = 2:10;
num_trials = 10;
threshold = 1e-4;

% Define penalty parameter values
lambda_values = [15, 20, 25, 30];

% Initialize variables to store results
best_wcss = zeros(length(Krange), 1); % Best WCSS for each k
f_k_lambda = zeros(length(Krange), length(lambda_values)); % f(k, λ) for each k and λ

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

% Calculate f(k, λ) for each value of λ
for lambda_idx = 1:length(lambda_values)
    lambda = lambda_values(lambda_idx);
    
    % Calculate f(k, λ) = WCSS_k-means + λk for each k in Krange
    f_k_lambda(:, lambda_idx) = best_wcss + lambda * Krange';
end

% Plot f(k, λ) as a function of k for each λ
figure;
hold on;
for lambda_idx = 1:length(lambda_values)
    plot(Krange, f_k_lambda(:, lambda_idx), '-o', 'LineWidth', 2, 'DisplayName', ['\lambda = ', num2str(lambda_values(lambda_idx))]);
end
title('f(k, \lambda) vs. Number of Clusters (k)');
xlabel('Number of Clusters (k)');
ylabel('f(k, \lambda)');
legend('Location', 'Best');
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
