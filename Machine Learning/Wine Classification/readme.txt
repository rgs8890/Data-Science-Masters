For this Machine Learning Project, I compared the KNN and Random Forest models in the MultiClassification Wine Problem. 
There are 9 distinct classes to predict from, however all the wines contain ratings from 3-8. We employ imblean and use synthetic sampling
to account for the class imbalance. We iterate through the number of nearest neighbours (k) as the hyperparameter for the KNN model during optimisation, and the maximum depth of the tree, the number of node splits and the number of tress in the Random Forest Model.

Random Forest Model Optimal Hyper-Parameters include:
- Minimum Leaf Size of 5
- Number of Predictors at 3
- Maximum Number of Splits at 40

KNN Model Optimal Hyper-Parameters include:
- K = 1
