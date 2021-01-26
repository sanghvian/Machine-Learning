function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
    % Basically we will find distance of each training example (from 300) from 3
    % clusters and store the results in a 300x3 mtx(called allDiff). Idx will basically
    % be a vector such that it's ith elem just stores the index of the
    % minimum column value for the ith row in the allDiff mtx

    m = size(X,1);      
    K = size(centroids,1);
    
    allDist = zeros(m, K);
       
    for i=1:m
        for j=1:K
            allDist(i,j) = calculateDist(X(i,:),centroids(j,:));
        end
    end
        
    [minDis,idx] = min(allDist,[],2);
    
% =============================================================

end

