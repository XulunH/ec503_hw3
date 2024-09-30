% EC 503 - HW 3
% DP-Means starter code

clear, clc, close all;
rng('default'); % For reproducibility
defaultseed = rng;

%% Generate Gaussian data:
% Add code below:


%% Generate NBA data:
% Add code below:

% HINT: readmatrix might be useful here
% Store it in DATA
DATA = 
% Make sure to remove column headings

%% DP Means method:

% Parameter Initializations
LAMBDA = 0.15;
convergence_threshold = 1;
num_points = length(DATA);
total_indices = [1:num_points];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DP Means - Initializations for algorithm %%%
% cluster count
K = 1;

% sets of points that make up clusters
% L is a cell array and one cell of L contains points belonging to a single cluster
L = {}; 
L = [L [1:num_points]];

% Class indicators/labels=stores labels of all the data points
Z = ones(1,num_points);

% means MU=stores cluster means
MU = [];
MU = [MU; mean(DATA,1)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initializations for algorithm:
% converged = 0 => not converged, converged = 1 => converged
converged = 0;
t = 0;
while (converged == 0)
    t = t + 1;
    fprintf('Current iteration: %d...\n',t)
    
    %% Per Data Point:
    for i = 1:num_points
        
        %% CODE 1 - Calculate distance from current point to all currently existing clusters
        % Write code below here:
        
        %% CODE 2 - Find the distance to closest cluster. If the distance is more than LAMBDA start a new cluster else update Z(i)
        % Write code below here:

    end
    
    %% CODE 3 - Form new sets of points (clusters)
    % Write code below here:
    % One cell of L should contain points belonging to a single cluster
    
    %% CODE 4 - Recompute means per cluster
    % Write code below here:
    
    %% CODE 5 - Test for convergence: number of clusters doesn't change and means stay the same
    % Write code below here:
    
    %% CODE 6 - Plot final clusters after convergence 
    % Write code below here:
    
    if (converged)
        %%%%
    end    
end




