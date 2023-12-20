X_train = readtable('C:\Users\rgs88\OneDrive\Documents\Semester 1 Data Science MSc Projects\Machine Learning\ML_Retake\X_train.csv');
y_train = readtable('C:\Users\rgs88\OneDrive\Documents\Semester 1 Data Science MSc Projects\Machine Learning\ML_Retake\y_train.csv');
X_test = readtable('C:\Users\rgs88\OneDrive\Documents\Semester 1 Data Science MSc Projects\Machine Learning\ML_Retake\X_test.csv');
y_test = readtable('C:\Users\rgs88\OneDrive\Documents\Semester 1 Data Science MSc Projects\Machine Learning\ML_Retake\y_test.csv');

% Convert tables to arrays
X_train = table2array(X_train);
y_train = table2array(y_train);
X_test = table2array(X_test);
y_test = table2array(y_test);

% Train a simple KNN model
k = 5; % Set the number of neighbors
KNNModel = fitcknn(X_train, y_train, 'NumNeighbors', k);

