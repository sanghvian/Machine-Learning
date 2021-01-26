function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
% p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% There will basically be 2 computations

% Computation 1 :
% Starting Layer = 400 activation nodes
% Destination Layer = 25 activation nodes
% Inps = X[ones(5000,1) X] (5000*401) vector 
% Weights = Theta1 (25*401)
% Outputs = a_two (25*5000) vector (each row basically represents the calculated value of that particular activation node across 5000 examples)

biased_X = [ones(m,1) X];
a_two = sigmoid(Theta1*((biased_X)')); % 25*5000 

a_two_next = (a_two)'; % 5000*25

% Computation 2 :
% Starting Layer = 25 activation nodes
% Destination Layer = 10 activation nodes
% Inps = [ones(5000,1) a_two_next] (5000*26) vector 
% Weights = Theta2 (10*26)
% Outputs = a_three (10*5000) vector (each row basically represents the calculated value of that particular activation node across 5000 examples)

biased_a_two_next = [ones(m,1) a_two_next];
a_three = sigmoid(Theta2*((biased_a_two_next)')); %10*5000

% Now, we have the final matrix : a_three from which we can easily extract
% all the classified predictions

a_three_next = a_three'; %5000*10
[~,p] = max(a_three_next, [], 2);

% =========================================================================


end
