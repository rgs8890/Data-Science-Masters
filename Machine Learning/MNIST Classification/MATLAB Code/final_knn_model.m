X_train = readtable('C:\Users\rgs88\OneDrive\Documents\Semester 1 Data Science MSc Projects\Machine Learning\ML_Retake\X_train.csv');
y_train = readtable('C:\Users\rgs88\OneDrive\Documents\Semester 1 Data Science MSc Projects\Machine Learning\ML_Retake\y_train.csv');
X_test = readtable('C:\Users\rgs88\OneDrive\Documents\Semester 1 Data Science MSc Projects\Machine Learning\ML_Retake\X_test.csv');
y_test = readtable('C:\Users\rgs88\OneDrive\Documents\Semester 1 Data Science MSc Projects\Machine Learning\ML_Retake\y_test.csv');

% Convert tables to arrays
X_train = table2array(X_train);
y_train = table2array(y_train);
X_test = table2array(X_test);
y_test = table2array(y_test);

% Splitting the data into smaller number of training samples
num_samples_train = 5000; % Number of training samples
num_samples_test = 1000;  % Number of test samples

random_indices_train = randperm(size(X_train, 1), num_samples_train);
X_train_subset = X_train(random_indices_train, :);

y_train_subset = y_train(random_indices_train, :);

random_indices_test = randperm(size(X_test,1), num_samples_test);
X_test_subset = X_test(random_indices_test, :);

y_test_subset = y_test(random_indices_test, :);

%Final Model
final_model_knn = fitcknn(X_train_subset, y_train_subset, 'NumNeighbors', 1);

%Testing and Evaluation
%Testing model
y_pred_labels = predict(final_model_knn, X_test_subset);

% Evaluate the model
accuracy = sum(y_pred_labels == y_test_subset) / numel(y_test_subset);

% Display accuracy
disp(['Test Accuracy: ', num2str(accuracy * 100), '%']);
