%Final Model after hyper-parameter tuning
DT_Final_Model = fitctree(X_train, ...
                       Y_train, ...
                       "MaxNumSplits", 17, ...
                       "MinLeafSize", 1, ...
                       "SplitCriterion", "deviance");

DT_fm_chart = confusionchart(Y_test, y_predict);

DT_Mf = confusionmat(Y_test, y_predict);

diagonal = diag(DT_Mf);
sum_of_rows = sum(DT_Mf, 2);

precision = diagonal ./ sum_of_rows;
overall_precision = mean(precision);

sum_of_columns = sum(DT_Mf, 1);

recall = diagonal ./ sum_of_columns';
overall_recall = mean(recall);

f1_score = 2* ((overall_precision*overall_recall)/ (overall_precision+overall_recall));
