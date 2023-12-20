%This is my default KNN Model
%The hyper=parameters used are default values
Initial_Model_KNN = fitcknn(X_train, ...
                            Y_train, ...
                            "NumNeighbors", 5);

KNN_fm_chart = confusionchart(Y_test, y_predict);

KNN_M = confusionmat(Y_test, y_predict);

diagonal = diag(KNN_M);
sum_of_rows = sum(KNN_M, 2);

precision = diagonal ./ sum_of_rows;
overall_precision = mean(precision);

sum_of_columns = sum(KNN_M, 1);

recall = diagonal ./ sum_of_columns';
overall_recall = mean(recall);

f1_score = 2* ((overall_precision*overall_recall)/ (overall_precision+overall_recall));