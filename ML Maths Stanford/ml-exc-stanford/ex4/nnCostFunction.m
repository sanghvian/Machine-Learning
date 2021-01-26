function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the COST and GRADINET of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: FORWARD PROPAGATION in the neural network to return the COST in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.mlx

a1 = X; % 5000*400
a1 = [ones(m,1) a1]; %5000*401
z2 = a1*(Theta1'); % 5000*25
a2 = sigmoid(z2); % 5000*25
a2 = [ones(m,1) a2]; %5000*26
z3 = a2*(Theta2'); %5000*10
a3 = sigmoid(z3); % 5000*10

h = a3;

all_y = zeros(size(h)); % 5000*10
for i=1:m
    all_y(i,y(i)) = 1;
end
when_y_is_one = all_y.*(log(h)); % 5000*10
when_y_is_zero = (1-all_y).*(log(1-h)); % 5000*10
J = -(1/m)*sum(sum(when_y_is_zero + when_y_is_one));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.

X_t = [ones(m,1) X]; % Created a useful transpose of X here itself so that we don't repeat the same step again and again in loop
for i=1:m
    % Forward Prop to find all a_l
    a1 = X_t(i,:); % 1*401
    a1 = a1'; % 401*1
    % Theta1 => 25*401
    z2 = Theta1*a1; % 25*1    
    a2 = sigmoid(z2); % 25*1
    
    a2 = [1; a2]; % 26*1
    % Theta2 = 10*26
    z3 = Theta2*a2; % 10*1
    a3 = sigmoid(z3); % 10*1
    
    % Back prop to compute grad
    % all_y => 5000*10
    curr_y = (all_y(i,:))'; % 10*1
    delta3 = a3 - curr_y; % 10*1
    
    z2 = [1; z2]; % 26*1, to use for element-wise multiplication after sigmoidGradient
    delta2 = (Theta2'*delta3).*sigmoidGradient(z2); % 26*1
    delta2 = delta2(2:end,:); % 25*1
    
    Theta2_grad = Theta2_grad + (delta3*(a2')); % 10*26
    Theta1_grad = Theta1_grad + (delta2*(a1')); % 25*401
end

Theta1_grad = (1/m)*Theta1_grad;
Theta2_grad = (1/m)*Theta2_grad;

% Part 3: REGULARIZATION with the cost function and gradients.

reg_theta_1 = Theta1(:,2:end); % 25*400
reg_theta_2 = Theta2(:,2:end); % 10*25
reg_term = (lambda/(2*m))*(sum(sum((reg_theta_1).^2)) + sum(sum((reg_theta_2).^2)));
J = J + reg_term;

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
