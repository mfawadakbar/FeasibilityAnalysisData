function W = randInitializeWeights(L_in, L_out)


%L_in and L_out are the incomming and outgoing connections
W = zeros(L_out, 1 + L_in);



% Randomly initialize the weights to small values
    
epsilon_init = 0.12; 
W = rand(L_out, 1 + L_in).*((2*epsilon_init)-epsilon_init);






% =========================================================================

end
