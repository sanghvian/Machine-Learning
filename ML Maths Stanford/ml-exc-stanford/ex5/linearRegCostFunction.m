function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% Basically, our required formula is :
% J = 1/2m (sum((h - y)^2) + lambda/2m(sum((theta)^2))

% For the unreg term
% X => 12x2, theta => 2x1
const1 = 1/(2*m);
h = X*theta; % h => 12x1, y => 12x1
diff = h-y; % diff => 12x1
sq = diff.^2; % sq => 12x1
sum1 = sum(sq);
unreg_term = const1*sum1;

% For the reg term
const2 = lambda/(2*m);
theta_reg = theta;
theta_reg(1,:) = 0;
sq_theta = theta_reg.^2;
sum2 = sum(sq_theta);
reg_term = const2*sum2;

J = unreg_term + reg_term;

% For calculating the grad term
% X => 12x2, X' => 2x12, diff => 12x1
% X'*diff => 2x1
% theta_reg => 2x1
grad = 2*const1*(X'*diff);
grad = grad + 2*const2*theta_reg;

% =========================================================================

grad = grad(:);

end
