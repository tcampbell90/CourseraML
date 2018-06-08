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

n = length(theta)
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


%Non reg : grad = (1/m)*(sigmoid(X*theta)-y)'*X

grad(1) = (1/m)*(sigmoid(X*theta)-y(:)'*X(: , 1)

for k = 2:n

grad(k) = (1/m)*(sigmoid(X*theta)-y)'*X(: , k)' + (lambda/m).*theta(k)
  
endfor


% Non Reg Cost: J = (1/m)*sum(-y'*log(sigmoid(X*theta))-(1-y)'*log(1-sigmoid(X*theta)))

J = (1/m)*sum(-y'*log(sigmoid(X*theta))-(1-y)'*log(1-sigmoid(X*theta)))-(lambda/(2*m)).*(theta(2:n)'*theta(2:n))


% =============================================================

end
