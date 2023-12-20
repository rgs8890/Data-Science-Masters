clc, clear all, format compact
Heart_Data = readtable("heartdata_cleaned.csv");

Heart_Data;

%Cross-Validation
%cv = cvpartition(size(Heart_Data, 1), 'K-Fold', 0.3);

%cv1 = cvpartition(size(Heart_Data,1), 'KFold', 5);
cv = cvpartition(size(Heart_Data,1), 'HoldOut', 0.3);

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

%Decision Tree
Mdl=fitctree(X_train, Y_train);
y_predict = predict(Mdl, X_test);
view(Mdl, 'Mode', 'graph');

%Accuracy
result = confusionmat(Y_test, y_predict);
chart = confusionchart(Y_test, y_predict);

%Optimizing Hyper-Parameters
Mdl1 = fitctree(X_train, Y_train, 'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct('Holdout', 0.3,...
    'AcquisitionFunctionName', 'expected-improvement-plus'));


%Hyper-Parameter Tuning of decison tree
maxMinLS = 20;
minLS = optimizableVariable('minLS', [1, maxMinLS], 'Type', 'integer');
numPTS = optimizableVariable('numPTS', [1, size(Heart_Data,2)-1], 'Type', 'integer');
hyperparametersDT = [minLS; numPTS];

%Selecting appropriate Tree Depth (controlling depth of the tree)
leaves = logspace(1,2,10);
rng('default')
N = numel(leaves);
err = zeros(N,1);
for n=1:N
    t = fitctree(X_train, Y_train, 'CrossVal', 'On', ...)
        'MinLeafSize', leaves(n));
end
plot(leaves, err);
xlabel('Minimum Leaf Size');
ylabel('Cross-Validated Error');