# Insurance-Fraud-Prediction
Insurance fraud prediction in R using class balancing, Random Forest, and validation-based threshold optimization.

## Model Selection

Several classification approaches were explored, including logistic regression, Random Forest, and XGBoost.

Because fraudulent claims represented a minority class, random undersampling was applied to the training data. The final model was selected based on its ability to balance overall accuracy with fraud-case recall.

A Random Forest classifier produced the strongest validation performance. The classification threshold was tuned on a separate validation set rather than using the default threshold of 0.50. The selected threshold was approximately 0.49 and maximized the project-specific score defined as:

`Final score = Accuracy × Recall`

The tuned model was then used to generate predictions for the holdout dataset.
