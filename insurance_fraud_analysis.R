# ============================================================
# Insurance Fraud Prediction
# Random Forest with Threshold Optimisation
# ============================================================
#
# This script combines the final modelling steps developed
# during the Business Analytics project:
#
#   1. Load the insurance claims data
#   2. Separate labelled and holdout observations
#   3. Split labelled data into training and validation sets
#   4. Apply random undersampling to the training set
#   5. Train a Random Forest model using ranger
#   6. Select the threshold that maximises Accuracy × Recall
#   7. Train the final model on all labelled observations
#   8. Generate predictions for the holdout sample
#
# Required packages:
# dplyr, ranger
#
# Expected project structure:
#
# insurance-fraud-prediction/
# ├── insurance_fraud_analysis.R
# └── data/
#     └── insurance_claims.csv
# ============================================================


# 1. Load packages ---------------------------------------------------------

library(dplyr)
library(ranger)


# 2. Load data -------------------------------------------------------------

data1 <- read.csv("data/insurance_claims.csv")

# Labelled observations are used for model development.
# Holdout observations are used for the final predictions.

train_data <- data1 %>%
  filter(holdout == 0)

test_data <- data1 %>%
  filter(holdout == 1)


# 3. Create training and validation sets ----------------------------------

set.seed(123)

train_index <- sample(
  nrow(train_data),
  size = 0.8 * nrow(train_data)
)

train_set <- train_data[train_index, ]
validation_set <- train_data[-train_index, ]


# 4. Random undersampling -------------------------------------------------

# Fraud cases form the minority class. The majority class is randomly
# sampled to create a balanced dataset for model training.

fraud_cases <- train_set %>%
  filter(fraud == 1)

nonfraud_cases <- train_set %>%
  filter(fraud == 0)

set.seed(42)

nonfraud_sample <- nonfraud_cases %>%
  sample_n(nrow(fraud_cases))

train_rus <- bind_rows(
  fraud_cases,
  nonfraud_sample
)

train_rus$fraud <- factor(
  train_rus$fraud,
  levels = c(0, 1)
)


# 5. Train Random Forest model --------------------------------------------

rf_ranger <- ranger(
  formula = fraud ~ .,
  data = train_rus %>%
    select(-holdout, -holdout_order),
  probability = TRUE,
  num.trees = 500,
  mtry = floor(sqrt(ncol(train_rus) - 3)),
  importance = "impurity"
)


# 6. Predict probabilities on validation set ------------------------------

predictors <- names(
  train_rus %>%
    select(-fraud, -holdout, -holdout_order)
)

validation_set_eval <- validation_set[, c(predictors, "fraud")]

validation_set_eval$predicted_prob <- predict(
  rf_ranger,
  data = validation_set_eval
)$predictions[, 2]


# 7. Tune classification threshold ---------------------------------------

# The project evaluation criterion is:
#
# Final Score = Accuracy × Recall
#
# Thresholds from 0.10 to 0.90 are evaluated on the validation set.

thresholds <- seq(
  from = 0.10,
  to = 0.90,
  by = 0.01
)

results <- data.frame(
  threshold = thresholds,
  accuracy = NA,
  recall = NA,
  final_score = NA
)

for (i in seq_along(thresholds)) {

  current_threshold <- thresholds[i]

  predicted_class <- ifelse(
    validation_set_eval$predicted_prob > current_threshold,
    1,
    0
  )

  accuracy <- mean(
    predicted_class ==
      as.numeric(as.character(validation_set_eval$fraud))
  )

  recall <- sum(
    predicted_class == 1 &
      validation_set_eval$fraud == 1
  ) / sum(validation_set_eval$fraud == 1)

  final_score <- accuracy * recall

  results[i, c("accuracy", "recall", "final_score")] <- c(
    accuracy,
    recall,
    final_score
  )
}

best_threshold <- results[
  which.max(results$final_score),
]

print(best_threshold)


# 8. Train final model on all labelled observations -----------------------

# Random undersampling is repeated using all labelled observations
# before training the model used for holdout predictions.

fraud_cases_full <- train_data %>%
  filter(fraud == 1)

nonfraud_cases_full <- train_data %>%
  filter(fraud == 0)

set.seed(42)

nonfraud_sample_full <- nonfraud_cases_full %>%
  sample_n(nrow(fraud_cases_full))

train_rus_full <- bind_rows(
  fraud_cases_full,
  nonfraud_sample_full
)

train_rus_full$fraud <- factor(
  train_rus_full$fraud,
  levels = c(0, 1)
)

final_rf_model <- ranger(
  formula = fraud ~ .,
  data = train_rus_full %>%
    select(-holdout, -holdout_order),
  probability = TRUE,
  num.trees = 500,
  mtry = floor(sqrt(ncol(train_rus_full) - 3)),
  importance = "impurity"
)


# 9. Generate holdout predictions ----------------------------------------

test_data_predict <- test_data[, predictors]

test_data$predicted_prob <- predict(
  final_rf_model,
  data = test_data_predict
)$predictions[, 2]

test_data$predicted_class <- ifelse(
  test_data$predicted_prob > best_threshold$threshold,
  1,
  0
)


# 10. Save final submission ------------------------------------------------

submission <- test_data %>%
  arrange(holdout_order) %>%
  pull(predicted_class)

writeLines(
  as.character(submission),
  "fraud_Lisianskaia.txt"
)


# 11. Optional outputs -----------------------------------------------------

# Save the threshold comparison table.

write.csv(
  results,
  "threshold_results.csv",
  row.names = FALSE
)

# Display variable importance from the final Random Forest model.

variable_importance <- data.frame(
  variable = names(final_rf_model$variable.importance),
  importance = as.numeric(final_rf_model$variable.importance)
) %>%
  arrange(desc(importance))

print(variable_importance)
