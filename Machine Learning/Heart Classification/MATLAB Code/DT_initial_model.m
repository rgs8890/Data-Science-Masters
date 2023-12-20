%Initial Model without any hyper-parameters specialized.
Initial_Model_DT = fitctree(X_train, ...
                            Y_train);

%Confusion Chart - TP, TN, FP, FN Values
DT_im_chart = confusionchart(Y_test, y_predict);

DT_M = confusionmat(Y_test, y_predict);

diagonal = diag(DT_M);
sum_of_rows = sum(DT_M, 2);

precision = diagonal ./ sum_of_rows;
overall_precision = mean(precision);

sum_of_columns = sum(DT_M, 1);

recall = diagonal ./ sum_of_columns';
overall_recall = mean(recall);

f1_score = 2* ((overall_precision*overall_recall)/ (overall_precision+overall_recall));