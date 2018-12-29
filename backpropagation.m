tic
% Training variables
epochs = 10;
hidden_units = 50;
learning_rate = .00005;

% Load the training and test data:
% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
test_images = loadMNISTImages('t10k-images-idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
 
% We are using display_network from the autoencoder code
train_input = images(:,1:60000); % Show the first 100 images
train_output = labels(1:60000);
test_input = test_images(:,1:4999); % Show the first 100 images
test_output = test_labels(1:4999);
test_input2 = test_images(:,5000:10000); % Show the first 100 images
test_output2 = test_labels(5000:10000);

input_size = size(train_input);
test_size = size(test_input);
% Determine how many character patterns & pixels in each character we have
inputs = size(train_input,1);  % = 784
patterns = size(train_input,2); % = 60k
testpatterns = size(test_input,2); % = 60k

% Target outputs
outZero = [1,0,0,0,0,0,0,0,0,0];
outOne = [0,1,0,0,0,0,0,0,0,0];
outTwo = [0,0,1,0,0,0,0,0,0,0];
outThree = [0,0,0,1,0,0,0,0,0,0];
outFour = [0,0,0,0,1,0,0,0,0,0];
outFive = [0,0,0,0,0,1,0,0,0,0];
outSix = [0,0,0,0,0,0,1,0,0,0];
outSeven = [0,0,0,0,0,0,0,1,0,0];
outEight = [0,0,0,0,0,0,0,0,1,0];
outNine = [0,0,0,0,0,0,0,0,0,1];

outputMatrix = [outZero;outOne;outTwo;outThree;outFour;outFive;outSix;outSeven;outEight;outNine];

% Create target values for the training and test outputs
train_target = zeros(10,patterns);
for num = 1:patterns
   train_target(:,num) = train_target(:,num) + outputMatrix(:,train_output(num)+1); 
end
test_target = zeros(10,testpatterns);
for num = 1:testpatterns
   test_target(:,num) = test_target(:,num) + outputMatrix(:,test_output(num)+1); 
end
test_target2 = zeros(10,testpatterns);
for num = 1:testpatterns
   test_target2(:,num) = test_target2(:,num) + outputMatrix(:,test_output2(num)+1); 
end

% Create a random weight matrixs and biases
W1 = randn(hidden_units,inputs);   % input to hidden W matrix
W2 = randn(10,hidden_units); % hidden to output W matrix
b1 = rand(hidden_units,1); % bias for input to hidden
b2 = rand(10,1);     % bias for hidden to output

% Mean squared error arrays for total epoch training
mse = zeros(1,epochs);   % array holding the mse for one epoch
mse_test = zeros(1,epochs);   % array holding the mse for one epoch
mse_test2 = zeros(1,epochs);   % array holding the mse for one epoch

% Main loop - In one epoch we train the network using backpropagation
for epoch_iteration = 1:epochs
    squared_errors = zeros(1,patterns); % error values for training data for one epoch
    squared_testerrors = zeros(1,testpatterns); % error values for first 5k normal images in one epoch
    squared_testerrors2 = zeros(1,testpatterns); % error values for remaining 5k distorted images in one epoch
    
    for char = 1:patterns % Forward pass
        % Compute the layer output with logsig
        a1 = logsig(W1*train_input(:,char)+b1); %output of hidden
        a2 = logsig(W2*a1+b2); %output of network output

        % Compute the error e
        e = train_target(:,char) - a2;
          % Get total squared error for all pattern/characters
        squared_errors(char) = sum(e.^2);
    end
    for char = 1:testpatterns 
        % Compute the layer output with logsig
        a1 = logsig(W1*test_input(:,char)+b1); %output of hidden
        a2 = logsig(W2*a1+b2); %output of network output

        % Compute the error e
        e = test_target(:,char) - a2;

        % Get total squared error for all pattern/characters
        squared_testerrors(char) = sum(e.^2);
        
        a1 = logsig(W1*test_input2(:,char)+b1); %output of hidden
        a2 = logsig(W2*a1+b2); %output of network output

        % Compute the error e
        e = test_target(:,char) - a2;

        % Get total squared error for all pattern/characters
        squared_testerrors2(char) = sum(e.^2);

    end    
    for char = 1:patterns % Backward pass
        % Compute the layer output with logsig
        a1 = logsig(W1*train_input(:,char)+b1); %output of hidden
        a2 = logsig(W2*a1+b2); %output of network output

        % Compute the error e
        e = train_target(:,char) - a2;

        % Propagate the sensitivites backward through the network
        s2 = -2.*diag(ones(size(a2))-a2.*a2)*e;
        s1 = diag(ones(size(a1))-a1.*a1)*W2'*s2;

        % Update the weights and biases
        W1 = W1 - learning_rate*s1*train_input(:,char)';
        W2 = W2 - learning_rate*s2*a1';
        b1 = b1 - learning_rate.*s1;
        b2 = b2 - learning_rate.*s2;
    end
    
    % Get total mean squared error for single epoch
    mse(epoch_iteration) = mean(squared_errors);
    mse_test(epoch_iteration) = mean(squared_testerrors);
    mse_test2(epoch_iteration) = mean(squared_testerrors2);
end

% Graph the total squared error
plot(1:epochs, mse, 1:epochs, mse_test, 1:epochs, mse_test2);
title('Backpropagation Part 2: MNIST');
xlabel('Epochs');
ylabel('Mean Squared Error');
legend('Training', 'Test Images', 'Distorted Test Images');

toc