X_train = readtable('\X_train.csv');
Y_train = readtable('\Y_train.csv');
X_test = readtable('\X_test.csv');
Y_test = readtable('\Y_test.csv');

% Convert tables to arrays
X_train = table2array(X_train);
Y_train = table2array(Y_train);
X_test = table2array(X_test);
Y_test = table2array(Y_test);

% Train Final KNN Model
final_model_knn = fitcknn(X_train, Y_train, 'NumNeighbors', 1);

% Predict on the test data
y_pred = predict(final_model_knn, X_test);