function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.   

% Basically, I need to use X and the currently given optimized theta to
% come up with a set of output predictions where each prediction points to 1 of 10 (as num_labels = 10) possible output classes

% On doing - all_theta*(X') (where X (5000*401) and all_theta (10*401))
% we get for each training example (out of 5000), i.e. for each set of inputs(401) and weights(401) we basically end up
% getting output vector such that it has, as its rows, probabilities that
% the output is actually the class represented by that row. Out of these,
% whichever row has the maximum probability is chosen to be the output
% class i.e we basically end up choosing that particular elem's index.
% p is basically a 5000*1 matrix where each row is basically a no. between
% 1 and 10 showing that this training eg (outta 5000) translates to an
% output belonging to this class

outputs = X*((all_theta)'); % gives back a 5000*10 vector

% our p vector is basically corresponding to the indexes holding the max of
% each single row

[~,p] = max(outputs, [], 2);


% =========================================================================


end
