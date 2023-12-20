X_train = readtable('\X_train.csv');
y_train = readtable('\Y_train.csv');
X_test = readtable('\X_test.csv');
y_test = readtable('\Y_test.csv');

% Convert tables to arrays
X_train = table2array(X_train);
Y_train = table2array(Y_train);
X_test = table2array(X_test);
Y_test = table2array(Y_test);

% Train RF Final Model
numTrees = 100;
minLeafSize = 5;
numPredictorsToSample = 3;
maxNumSplits = 40;

final_rf_model = TreeBagger(numTrees, X_train, Y_train, ...
                            'Method', 'classification', ...
                            'MinLeafSize', minLeafSize, ...
                            'NumPredictorsToSample', numPredictorsToSample, ...
                            'MaxNumSplits', maxNumSplits);

% Test the SVM Model
% Predict on the test data
y_pred = predict(final_rf_model, X_test);