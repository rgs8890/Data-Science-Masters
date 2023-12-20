clear all;
clc;
% Loading Pre-Processed Data
bank_data = readtable("C:\Users\rgs88\OneDrive\Documents\Semester 2 Data Science Masters\Neural Computing\bank_data_preprocessed.csv");

%Setting input and target variables
X = table2array(bank_data(:, ["Age", "Balance", "IsActiveMember", "Gender"]));
y = table2array(bank_data(:, "Exited"));

%Split the dataset into training and test datasets
cvp = cvpartition(size(bank_data,1),'HoldOut', 0.3);
train_idx = cvp.training;
test_idx = cvp.test;

% Get the training and test data
X_train = X(train_idx, :);
y_train = y(train_idx);
X_test = X(test_idx, :);
y_test = y(test_idx);

%Creating the Neural Network

%Defining the size of the hidden layer
hiddenSize = 10;

%Defining the activation function
activationFunction = 'tansig';

%Define the learning rate
learning_rate = 0.01;

%Creating the neural network
net = patternnet(hiddenSize);
net.layers{1}.transferFcn = activationFunction;
net.trainParam.lr = learning_rate;

%Setting the training parameters
net.trainParam.epochs = 100;
net.trainParam.time = Inf;

%Training the neural network on the training dataset
r = size(X_train);
n = size(y_train);
q = sum(isnan(X_train)) ;% should return 0 if there are no missing values
s = sum(isnan(y_train)) ;% should return 0 if there are no missing values

net = train(net, X_train, y_train);

%Predicting the output on the test dataset
test_outputs_predicted = sim(net, X_test);

% Convert the output to the actual class labels
[~, actual_labels] = max(y_test);
[~, predicted_labels] = max(test_outputs_predicted);

% Calculate the test accuracy
test_accuracy = sum(actual_labels == predicted_labels)/length(actual_labels);

% Plot the test accuracy
bar(test_accuracy);
title('Test Accuracy');
xlabel('Test Dataset');
ylabel('Accuracy');

train_loss = [];
train_acc = [];

train_speed = zeros(1, net.trainParam.epochs);
for i = 1:net.trainParam.epochs
    tic;
    net = train(net, X_train, y_train);
    train_speed(i) = toc;
end

test_outputs_predicted = sim(net, X_test);
[~,actual_labels] = max(y_test);
[~, predicted_labels] = max(test_outputs_predicted);
test_accuracy = sum(actual_labels == predicted_labels)/length(actual_labels);

% Plot the training speeds
plot(train_speed);
title('Training Speed');
xlabel('Epochs');
ylabel('Time (s)');
