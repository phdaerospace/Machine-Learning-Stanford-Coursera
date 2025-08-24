function [J grad] = nnCostFunction(nn_params, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

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
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%% Forward propagation

% Implementation specific to this problem

% % Append +1 to the first layer i.e. the inpuit layer fopr all the examples
% % so that the X has 401 columns instead of 400
% bias1 = ones(5000,1);
% Xapp = [bias1 X];
%
% % Second hidden layer no of units calculated from first layer = 25
%
% % Second hidden layer extra unit +1 added later for calculation for third
% % layer
%
% % Theta1 dimension is 25*401
%
% Z12 = (Theta1*Xapp')';
%
% % for eg = 1:5000
% %     Z12Check(:,eg) = Theta1*Xapp(eg,:)';
% % end
% %  Z12CheckF = Z12Check';
%
%  % To calculate units a1,a2,...a25 we need to find the
% % second layer a1 = g(theta*x') = 1/(1+exp(Z12))
% for eg = 1:5000
%     for unit = 1: 25
%         temp1 = exp(-Z12(eg,unit));
%         a_lay2(eg,unit) = 1/(1+temp1);
%     end
% end
%
% % From layer 2 to layer 3
% % First append +1 bias unit to the a_lay2
% a_lay2_app = [bias1 a_lay2];
%
% % Third layer no of units = 10
% % Theta2 dimension is 10*26
%
% Z23 = (Theta2*a_lay2_app')';
%
%
% % To calculate the hyupothesis for all examples we need to find
% for eg = 1:5000
%     for unit = 1: 10
%         temp1 = exp(-Z23(eg,unit));
%         a_lay3(eg,unit) = 1/(1+temp1);
%     end
% end
%
% HTheta = a_lay3;
%
%
%
%
% % To calculate the cost function using given Theta
% tot = 0;
% unit_tot=0;
%
%
% % This also works
%
% % temp2 = eye(10);
% % y0 = temp2(:,10);
% % yr = temp2(:,1:9);
% % Yf = [y0 yr];
%
% % for eg = 1:5000
% %     for unit = 1: 10
% %         if y(eg)==10
% %             YU = Yf(:,1);
% %         elseif y(eg)==1
% %             YU = Yf(:,2);
% %         elseif y(eg)==2
% %             YU = Yf(:,3);
% %         elseif y(eg)==3
% %             YU = Yf(:,4);
% %         elseif y(eg)==4
% %             YU = Yf(:,5);
% %         elseif y(eg)==5
% %             YU = Yf(:,6);
% %         elseif y(eg)==6
% %             YU = Yf(:,7);
% %         elseif y(eg)==7
% %             YU = Yf(:,8);
% %         elseif y(eg)==8
% %             YU = Yf(:,9);
% %         elseif y(eg)==9
% %             YU = Yf(:,10);
% %         end
% %
% %         term11 = -YU(unit)*log(HTheta(eg,unit)) ;
% %         term22 = -(1-YU(unit))*log(1-HTheta(eg,unit)) ;
% %         %unit_tot = term11+term22+unit_tot;
% %         tot = term11+term22+tot;
% %
% %     end
% %     eg_total(eg,1)=unit_tot;
% %     %unit_tot=0;
% % end
%
% % Better code i belive
% Y1 = eye(10);
%
% for eg = 1:5000
%     for unit = 1: 10
%         for label = 1:10
%             if y(eg) == label
%                 YU = Y1(:,label);
%             end
%         end
%
%         term11 = -YU(unit)*log(HTheta(eg,unit)) ;
%         term22 = -(1-YU(unit))*log(1-HTheta(eg,unit)) ;
%         tot = term11+term22+tot;
%
%     end
% end
%
% cost = tot/5000;
% J = cost;

%% Making a generalized implementation

% no of examples - m
[m,n] = size(X);
% no of pixels in an image - n
% no of units in layer 1 = n

% Append +1 to the first layer i.e. the input layer for all the examples
% so that the X has 401 columns instead of 400
bias1 = ones(m,1);
Xapp = [bias1 X];

units_ip = input_layer_size+1;

% First hidden layer (second layer) no of units calculated from first layer

% Second layer shud have extra unit +1 and is added later for calculation for third
% layer

% Theta1 dimension is 25*401 (hidden_layer_size * input_layer_size)

Z12 = (Theta1*Xapp')';

% To calculate units a1,a2,...a25 we need to find the
% second layer a1 = g(theta*x') = 1/(1+exp(Z12))

for eg = 1:m
    for unit = 1: hidden_layer_size
        temp1 = exp(-Z12(eg,unit));
        a_lay2(eg,unit) = 1/(1+temp1);
    end
end

% From layer 2 to layer 3
% First append +1 bias unit to the a_lay2
a_lay2_app = [bias1 a_lay2];

units_hl = hidden_layer_size+1;

% Third layer no of units = 10 (o/p layer or no of labels)
% Theta2 dimension is 10*26 (num_labels * units_hl)

Z23 = (Theta2*a_lay2_app')';

units_op = num_labels;
% To calculate the final layer hyupothesis for all examples we need to find
for eg = 1:m
    for unit = 1: num_labels
        temp1 = exp(-Z23(eg,unit));
        a_lay3(eg,unit) = 1/(1+temp1);
    end
end

HTheta = a_lay3;


% To calculate the cost function using given Theta without regularization
tot = 0;

Y1 = eye(num_labels);

for eg = 1:m
    for unit = 1: units_op
        for label = 1:num_labels
            if y(eg) == label
                YU = Y1(:,label);
            end
        end

        term11 = -YU(unit)*log(HTheta(eg,unit)) ;
        term22 = -(1-YU(unit))*log(1-HTheta(eg,unit)) ;
        tot = term11+term22+tot;

    end
end

cost = tot/m;
J = cost;

%%  Cost function with regularization
% Theta1 size = 25*401 units in hidden layer (w/o bias)* units in ip layer
% (with bias)
% Theta2 size = 10*26 units in op layer w/o bias * units in hl with bias

% Comment this before submission
% lambda = 1;

com = lambda/(2*m);
term1 = 0;
term2 = 0;

for j = 1:hidden_layer_size
    for k = 2:input_layer_size+1
        term1 = term1 + (Theta1(j,k))^2;
    end
end

for j = 1:num_labels
    for k = 2:hidden_layer_size+1
        term2 = term2 + (Theta2(j,k))^2;
    end
end

reg = com * (term1+term2);

J = cost + reg;

%
%% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

% Delta for o/p layer

DELTA1 = zeros(size(Theta1));
DELTA2 = zeros(size(Theta2));

for t = 1:m
    for label = 1:num_labels
        if y(t) == label
            YU = Y1(:,label);
        end
    end

    % Step2
    delta_lay3 = a_lay3(t,:)' - YU;

    % Step 3
    %   delta_lay2 = (Theta2' * delta_lay3) .* sigmoidGradient(Z23(t,:));
    a2_vec = [1; a_lay2(t,:)'];
    a2_vec_size = size(a2_vec);
    % Theta2 first column is for bias units, to find Thtea2 without bias
    Theta2_wo_bias = Theta2(:,2:end);

    delta_lay2 = (Theta2' * delta_lay3) .* (a2_vec .* (ones(a2_vec_size(1),1)-a2_vec));

    % Step 4
    %     Calulating DELTA for first and swecond layer according to the formula
    %     DELTA = DELTA + a^l_j delta^(l+1)_j
    a_lay1 = Xapp(t,:)';
    a1_vec = a_lay1;

    %     temp2 = delta_lay2(2:end) * a1_vec';
    %     DELTA1 = DELTA1 + temp2 ;
    %
    %     temp3 = delta_lay3 * a2_vec(2:end)';
    %     DELTA2 = DELTA2 + temp3;

    temp2 = delta_lay2(2:end) * a1_vec';
    DELTA1 = DELTA1 + temp2 ;

    temp3 = delta_lay3 * a2_vec';
    DELTA2 = DELTA2 + temp3;



end

    DELTA1_grad = DELTA1/m;
    DELTA2_grad = DELTA2/m;


    DELTA1_grad(:,2:end) = DELTA1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end) ;
    DELTA2_grad(:,2:end) = DELTA2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end) ;


    Theta1_grad = DELTA1_grad ;
    Theta2_grad = DELTA2_grad ;



kk = 0;









%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
