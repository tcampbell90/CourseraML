function [theta, J_history] = gradientDescent(X, y, theta, alpha, iterations)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(iterations, 1);
%theta = zeros(2,1) ;


  for iter = 1:iterations

      % ====================== YOUR CODE HERE ======================
      % Instructions: Perform a single gradient step on the parameter vector
      %               theta.
      %
      % Hint: While debugging, it can be useful to print out the values
      %       of the cost function (computeCost) and gradient here.
      %
    
    % for k = 1:n
      % theta(k) = theta(k)-(alpha/m)*(X*theta - y)'*X(:, k)
    % end

    theta = theta - ((alpha/m)*(X*theta - y)'*X )'

      % ============================================================

      % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

    if iter == 1
      continue;
    elseif J_history(iter) < J_history(iter-1);
      continue;
    elseif J_history(iter) - J_history(iter-1) < 10^(-4)
      break;
    elseif (J_history(iter) == 0)
      break;
    endif;
    
    alpha = alpha + 0.001;
    
  endfor

plot(J_history)
  
end
