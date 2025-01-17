function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network


% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2')
p = h2(1);
% [dummy, p] = max(h2, [], 2);

% =========================================================================


end
