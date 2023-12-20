%Final Model
%After optimizing, we have our final KNN Model
KNN_Final_Model = fitcknn(X_train, ...
                          Y_train, ...
                          "NumNeighbors", 28, ...
                          "Distance", "cityblock", ...
                          "Standardize", 1 ...
                          );


KNN_fm_chart = confusionchart(Y_test, y_predict);

KNN_Mf = confusionmat(Y_test, y_predict);

diagonal1 = diag(KNN_Mf);
sum_of_rows1 = sum(KNN_Mf, 2);

precision1 = diagonal1 ./ sum_of_rows1;
overall_precision1 = mean(precision1);

sum_of_columns1 = sum(KNN_Mf, 1);

recall1 = diagonal ./ sum_of_columns1';
overall_recall1 = mean(recall);

f1_score1 = 2* ((overall_precision1*overall_recall1)/ (overall_precision1+overall_recall1));

