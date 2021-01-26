function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
% Basically, now that data points are attached to a centroid, we gotta
% compute what is gonna be the new mean of all these data points
dim = size(centroids,2);
for i=1:K
    % We will note down index of each element in idx (300x1) and using the value stored at that index, we
    % will find the corresponding examples in X(300x2) and then
    idxsOfCurrentCentroid = find(idx == i); % a row vec containing as many elems as many data points allotted to current centroid
    s = size(idxsOfCurrentCentroid,1);
    coordsOfAllottedDataPoints = zeros(s,dim);
    for j=1:s   
        coordsOfAllottedDataPoints(j,:) = X(idxsOfCurrentCentroid(j),:);
    end
    centroids(i,:) = sum(coordsOfAllottedDataPoints,1)./s;
end

centroids;

% =============================================================


end

