% EC 503 - HW 3
% K-Means starter code

clear, clc, close all;
rng('default'); % For reproducibility
defaultseed = rng;

%% Generate Gaussian data:
% Add code below:
% HINT: mvnrnd might be useful here

%% Generate NBA data:
% Add code below:

% HINT: readmatrix might be useful here
% Store it in DATA
DATA = 
% Make sure to remove column headings

% Problem 3.2(f): Generate Concentric Rings Dataset using
% sample_circle.m provided to you in the HW 3 folder on Blackboard.

%% K-Means implementation
% Add code below

K =
MU_init = 

MU_previous = MU_init;
MU_current = MU_init;

% initializations
labels = ones(length(DATA),1);
% converged = 0 => not converged, converged = 1 => converged
converged = 0;
iteration = 0;
convergence_threshold = 0.025;

% precompute term(s) of euclidean distance common to all iterations

while (converged==0)
    iteration = iteration + 1;
    fprintf('Iteration: %d\n',iteration)

    %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
    % While calculating the distance, use the same trick as in HW 2.5(e)
    % to avoid inner for loop.
    % Write code below here:
    
    %% CODE - Mean Updating - Update the cluster means
    % Write code below here:
    
    %% CODE 4 - Check for convergence 
    % Write code below here:
    
    if (something_here < convergence_threshold)
        converged=1;
    end
    
    %% CODE 5 - Plot clustering results if converged:
    % Write code below here:
    if (converged == 1)
        fprintf('\nConverged.\n')
        
        
       
        
        %% If converged, get WCSS metric
        % Add code below
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end



