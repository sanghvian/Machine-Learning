function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);
y_is_zero = y.*(log(h));
y_is_one = (1-y).*log(1-h);
const1 = 1/m;
const2 = lambda/(2*m);
theta_sq = (theta(2:end,:)).^2;
reg_term = const2*theta_sq;

J = -const1*(sum(y_is_zero + y_is_one)) + sum(reg_term);

X_zero = X(:,1);
theta_zero = theta(1,:);
h_zero = sigmoid(X_zero*theta_zero);
grad_zero = const1*((X_zero')*(h- y));

X_rest = X(:,2:end);
theta_rest = theta(2:end,:);
h_rest = sigmoid(X_rest*theta_rest);
deriv_reg_term = 2*const2*(theta_rest);
grad_rest  = const1*((X_rest')*(h - y)) + deriv_reg_term;

grad = [grad_zero; grad_rest];
% =============================================================

end
