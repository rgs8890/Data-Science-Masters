clc, clear all, format compact;
Heart_Data = readtable("heartdata_cleaned.csv");
Heart_Data;

%Cross-Validation
cv = cvpartition(size(Heart_Data, 1), 'Holdout', 0.3);

%Train Test Split
data_train = Heart_Data(cv.training, :);
data_test = Heart_Data(cv.test, :);
X_train = data_train(:, 1:11);
Y_train = data_train(:, 12);
X_test = data_test(:, 1:11);
Y_test = data_test(:, 12);

%Standardize of Data, Tables
X_train = normalize(X_train);
X_train = table2array(X_train);

X_test = normalize(X_test);
X_test = table2array(X_test);

Y_train = normalize(Y_train);
Y_train = table2array(Y_train);

Y_test = normalize(Y_test);
Y_test = table2array(Y_test);

%KNN Model
Mdl=fitcknn(X_train, Y_train, 'NumNeighbors',5,'Standardize', 1);
y_predict = predict(Mdl, X_test);

%Confusion Matrix
result = confusionmat(Y_test, y_predict);
chart = confusionchart(Y_test, y_predict);

%Hyper-Parameter Optimization
%gridsearch
KMdl = fitcknn(X_train, Y_train, 'OptimizeHyperparameters', 'auto',...
               'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
               'expected-improvement-plus'));