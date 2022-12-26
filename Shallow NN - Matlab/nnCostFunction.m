function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION  the neural network cost function for a two layer
%neural network which performs classification


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




        X = [ones(m, 1) X];

        for i=1:m
             Y(i,:) = zeros(num_labels,1);%Hardcoding num_labels here to 10 cause me 5 hours
             Y(i,:) = y(i,:);
        end


        a1 = X;
        z2 = a1*Theta1'; %940x5 * 5x25 = 940x25
        a2 = sigmoid(z2);
        a2 = [ones(m,1) a2]; %940x26
        z3 = a2*Theta2';  % 940x26 * 26x2 = 940x2
        a3 = sigmoid(z3);  % 940x2
        H0 = log(a3);     % 940x2
        H1 = log(1-a3);   
        
        costterm = (Y.*H0)+((1-Y).*H1);
  
        
        J = -sum(sum(costterm,2))/m;
        
        T1 = Theta1(:,2:(input_layer_size+1));
        T2 = Theta2(:,2:(hidden_layer_size+1));
        T1=T1.^2;
        T2=T2.^2;
        regterm = sum(sum(T1,2)) + sum(sum(T2,1));
        regterm = (lambda/(2*m))*regterm; %Forgetting parenthasis around 2*m cause me 40 mins.
        
        J = J + regterm;


%__________________________Backpropagation Algorithm_______________________ 
%                 
%                 delta3 = a3 - Y;
%                 delta2 = (delta3*Theta2).*sigmoidGradient(z2);

G1 = zeros(size(Theta1));
G2 = zeros(size(Theta2));

for i = 1:m
	ra1 = X(i,:)'; % X' = 401x1
	rz2 = Theta1*ra1; % (25x401)x(401x1) = 25x1
	ra2 = sigmoid(rz2);
	ra2 = [1;ra2]; % 26x1
	rz3 = Theta2*ra2;
	ra3 = sigmoid(rz3);
	
    %err3 = deltas for the output layer containing 10 units
    %err2 = deltas for the hidden layer containing 25 units
    
	err3 = ra3 - Y(i,:)'; % err3 = 10x1
	temp = Theta2'*err3; % Theta2' = 26x10 // temp = 26x1
	err2 = temp(2:end,1).*sigmoidGradient(rz2);% (25x1).*(25x1) = 25x1
	
    % G1,G2 = capital delta the total discrepensy in the each node for all
    %examples added up.
	
    G1 = G1 + err2 * ra1'; % 25x1*1x401 = 25x401
	G2 = G2 + err3 * ra2'; % 10x1*1x26 = 10x26
end

Theta1_grad = G1 / m + lambda*[zeros(hidden_layer_size , 1) Theta1(:,2:end)] / m;
Theta2_grad = G2 / m + lambda*[zeros(num_labels , 1) Theta2(:,2:end)] / m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
