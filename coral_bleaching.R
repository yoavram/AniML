library(tidyverse)
library(randomForest)
library(ROCR)

# 1. Load the dataset
# Using show_col_types = FALSE to keep the output clean
df <- read_csv("global_bleaching_environmental.csv", show_col_types = FALSE)

# 2. Convert 'Bleaching_Level' to factor
# The original Bleaching_Level column contains 'nd' for many records. 
# We'll also check if there are multiple classes.
working_df <- df %>%
  filter(Bleaching_Level != "nd" & !is.na(Bleaching_Level))

# If there are fewer than 2 classes in Bleaching_Level, 
# we'll create them from Percent_Bleaching to ensure the Random Forest can run.
if (length(unique(working_df$Bleaching_Level)) < 2) {
  message("Insufficient classes in Bleaching_Level. Creating binary levels from Percent_Bleaching...")
  working_df <- df %>%
    mutate(Percent_Bleaching_num = as.numeric(na_if(Percent_Bleaching, "nd"))) %>%
    filter(!is.na(Percent_Bleaching_num)) %>%
    mutate(Bleaching_Level = case_when(
      Percent_Bleaching_num <= 10 ~ "Low",
      TRUE ~ "High"
    )) %>%
    mutate(Bleaching_Level = as.factor(Bleaching_Level))
} else {
  working_df <- working_df %>%
    mutate(Bleaching_Level = as.factor(Bleaching_Level))
}

# 3. Pre-process and select features
# Selecting environmental drivers
data_clean <- working_df %>%
  select(Bleaching_Level, Latitude_Degrees, Longitude_Degrees, Distance_to_Shore, 
         Turbidity, Cyclone_Frequency, Depth_m, Temperature_Mean, Windspeed) %>%
  mutate(across(-Bleaching_Level, ~na_if(as.character(.), "nd"))) %>%
  mutate(across(-Bleaching_Level, as.numeric)) %>%
  na.omit()

# 4. Split the data: 70% Train / 30% Test
set.seed(123)
train_idx <- sample(seq_len(nrow(data_clean)), size = 0.7 * nrow(data_clean))
train_data <- data_clean[train_idx, ]
test_data  <- data_clean[-train_idx, ]

# 5. Train baseline Random Forest on the imbalanced training data
rf_model <- randomForest(Bleaching_Level ~ ., data = train_data, importance = TRUE)

baseline_predictions <- predict(rf_model, test_data)
baseline_conf_matrix <- table(Observed = test_data$Bleaching_Level, Predicted = baseline_predictions)
baseline_high_recall <- baseline_conf_matrix["High", "High"] / sum(baseline_conf_matrix["High", ])

baseline_probabilities <- predict(rf_model, test_data, type = "prob")
high_probabilities <- tibble(High_Probability = baseline_probabilities[, "High"])

rocr_prediction <- prediction(
  predictions = baseline_probabilities[, "High"],
  labels = test_data$Bleaching_Level == "High"
)
pr_performance <- performance(rocr_prediction, measure = "prec", x.measure = "rec")
auc_pr <- performance(rocr_prediction, measure = "aucpr")@y.values[[1]]

default_threshold_predictions <- if_else(
  baseline_probabilities[, "High"] > 0.5,
  "High",
  "Low"
) %>%
  factor(levels = levels(test_data$Bleaching_Level))

default_threshold_conf_matrix <- table(
  Observed = test_data$Bleaching_Level,
  Predicted = default_threshold_predictions
)
default_threshold_precision <- default_threshold_conf_matrix["High", "High"] /
  sum(default_threshold_conf_matrix[, "High"])
default_threshold_recall <- default_threshold_conf_matrix["High", "High"] /
  sum(default_threshold_conf_matrix["High", ])

high_probability_plot <- ggplot(high_probabilities, aes(x = High_Probability)) +
  geom_histogram(binwidth = 0.05, fill = "steelblue", color = "white") +
  geom_vline(xintercept = 0.25, color = "firebrick", linetype = "dashed", linewidth = 1) +
  labs(
    title = "Predicted Probabilities for the High Class",
    x = "Predicted Probability of High",
    y = "Count"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  "high_probability_histogram.png",
  plot = high_probability_plot,
  width = 8,
  height = 5,
  dpi = 150
)

png("high_precision_recall_curve.png", width = 1200, height = 900, res = 150)
plot(
  pr_performance,
  main = "Precision-Recall Curve for High Bleaching",
  col = "steelblue",
  lwd = 2
)
points(
  x = default_threshold_recall,
  y = default_threshold_precision,
  pch = 19,
  col = "firebrick",
  cex = 1.2
)
text(
  x = default_threshold_recall,
  y = default_threshold_precision,
  labels = "Threshold = 0.5",
  pos = 4,
  col = "firebrick"
)
dev.off()

threshold_predictions <- if_else(
  baseline_probabilities[, "High"] > 0.25,
  "High",
  "Low"
) %>%
  factor(levels = levels(test_data$Bleaching_Level))

threshold_conf_matrix <- table(Observed = test_data$Bleaching_Level, Predicted = threshold_predictions)
threshold_high_recall <- threshold_conf_matrix["High", "High"] / sum(threshold_conf_matrix["High", ])

cat("Baseline Random Forest:\n")
print(rf_model)
cat("\nBaseline Confusion Matrix:\n")
print(baseline_conf_matrix)
cat(sprintf("Recall for High (baseline): %.4f\n\n", baseline_high_recall))

cat("High probability histogram saved to high_probability_histogram.png\n\n")
cat(sprintf("AUC-PR for High: %.4f\n", auc_pr))
cat("Precision-Recall curve saved to high_precision_recall_curve.png\n")
cat(sprintf(
  "Default threshold (0.5) point on PR curve: precision = %.4f, recall = %.4f\n\n",
  default_threshold_precision,
  default_threshold_recall
))
cat("Threshold-Moved Confusion Matrix (High if probability > 0.25):\n")
print(threshold_conf_matrix)
cat(sprintf("Recall for High (threshold 0.25): %.4f\n\n", threshold_high_recall))

if (threshold_high_recall > baseline_high_recall) {
  cat("Recall for High improved after threshold moving.\n\n")
} else if (threshold_high_recall < baseline_high_recall) {
  cat("Recall for High did not improve after threshold moving.\n\n")
} else {
  cat("Recall for High stayed the same after threshold moving.\n\n")
}

# 6. Downsample the majority class in the training data to balance learning
minority_size <- train_data %>%
  count(Bleaching_Level) %>%
  summarise(min_n = min(n)) %>%
  pull(min_n)

balanced_train_data <- train_data %>%
  group_by(Bleaching_Level) %>%
  slice_sample(n = minority_size) %>%
  ungroup()

rf_model_balanced <- randomForest(
  Bleaching_Level ~ .,
  data = balanced_train_data,
  importance = TRUE
)

balanced_predictions <- predict(rf_model_balanced, test_data)
balanced_conf_matrix <- table(Observed = test_data$Bleaching_Level, Predicted = balanced_predictions)
balanced_high_recall <- balanced_conf_matrix["High", "High"] / sum(balanced_conf_matrix["High", ])

cat("Balanced Random Forest (downsampled training data):\n")
print(rf_model_balanced)
cat("\nBalanced Confusion Matrix:\n")
print(balanced_conf_matrix)
cat(sprintf("Recall for High (balanced): %.4f\n\n", balanced_high_recall))

if (balanced_high_recall > baseline_high_recall) {
  cat("Recall for High improved after downsampling.\n")
} else if (balanced_high_recall < baseline_high_recall) {
  cat("Recall for High did not improve after downsampling.\n")
} else {
  cat("Recall for High stayed the same after downsampling.\n")
}

# 7. Plot Variable Importance for the balanced model
varImpPlot(rf_model_balanced, main = "Drivers of Coral Bleaching (Balanced RF)")