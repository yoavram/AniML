# 🪸 Session 2: Coral bleaching with R
**Objective:** Can we predict whether a coral reef will suffer "High" or "Low" bleaching based on environmental conditions?

**The Data:**
- **Source:** Global Coral Bleaching Database (subset).
- **Target:** `Bleaching_Level` (High vs. Low).
- **Features:** Water Temperature, Depth, Turbidity, Distance to Shore, Windspeed.
- Available from [Kaggle](https://www.kaggle.com/datasets/pnminh95/global-bleaching-environmental?resource=download&select=global_bleaching_environmental.csv) as a CSV file.

**The Mission:**

1. Download the data from Kaggle (see linke above).
2. Switch the Colab Runtime to **R**.
3. Upload the data file to Colab.

Use the Gemini Agent to:
3. Convert `Bleaching_Level` to a **Factor** (Crucial for classification in R!).
4. Train a **Random Forest** model.
5. Visualize **Variable Importance** (Which environmental factor is the "killer"?).

## Prompt

```
Write R code  using tidyverse and randomForest.

Load the dataset from global_bleaching_environmental.csv.

Convert the 'Bleaching_Level' column to a factor (classification).

Split the data into 70% training and 30% testing.

Train a Random Forest to predict 'Bleaching_Level' using all other columns.

Print the Confusion Matrix using table().

Plot the Variable Importance using varImpPlot() to see the drivers of bleaching.
```

## Teaching points

The "Factor" Trap: This is the #1 bug in R Machine Learning.

Explanation: "If you don't run as.factor(), the model will try to predict the average bleaching (Regression). We want to predict the category (Classification). Gemini knows this, but you must ask it specifically."

The "Formula" (~ .): Explain how elegant R is here. Bleaching ~ . means "Bleaching depends on everything." No need to list columns X = [...] like in Python.

Biology Check: Look at the varImpPlot.

Temperature should be at the top (MeanDecreaseGini).

Depth might be second (deeper = cooler/less light).

Ask: "Does this match what we know about physiology?" (Yes).


### The Accuracy Paradox.

The Trap: "If I told you my model is 96% accurate, would you buy it?" (Most will say yes).

The Reveal: "This model missed 67% of the severe bleaching events (346 missed out of 515). If we used this for conservation, the reef would die while the computer says 'Everything is fine'."

The Cause: Class Imbalance. There are ~15x more "Low" samples than "High". The model learned it can just guess "Low" and be right most of the time.

The Fix: Weighted Random Forest

The easiest fix in R's randomForest is to penalize mistakes on the rare class using classwt or sampsize.
Prompt:
```
The model has high accuracy but poor recall for 'High' bleaching because of class imbalance. Retrain the Random Forest model, but this time downsample the majority class to match the minority class size (or use classwt) to balance the learning. Compare the new Confusion Matrix to the old one. Did Recall for 'High' improve?
```

## Source code

```R
# 1. Install/Load Packages (Colab has them pre-installed usually, but good to check)
suppressPackageStartupMessages(library(tidyverse))
library(randomForest)

# 2. Load Data
url <- "https://raw.githubusercontent.com/yoavram/MarineML/main/data/coral.csv" # Replace with real URL
coral <- read_csv(url)

# 3. Data Prep
# In R, randomForest performs Regression if target is numeric, and Classification if Factor.
# We MUST convert to factor.
coral <- coral %>%
  mutate(Bleaching_Level = as.factor(Bleaching_Level)) %>%
  drop_na()

# 4. Split Data
set.seed(42)
sample_size <- floor(0.7 * nrow(coral))
train_ind <- sample(seq_len(nrow(coral)), size = sample_size)

train <- coral[train_ind, ]
test <- coral[-train_ind, ]

# 5. Train Model
# Formula: Target ~ . (means "predicted by everything else")
rf_model <- randomForest(Bleaching_Level ~ ., data = train, ntree = 100)

# 6. Evaluate
predictions <- predict(rf_model, test)
print("Confusion Matrix:")
print(table(Predicted = predictions, Actual = test$Bleaching_Level))

# 7. Check Accuracy
accuracy <- sum(predictions == test$Bleaching_Level) / nrow(test)
print(paste("Accuracy:", round(accuracy, 2)))

# 8. Variable Importance Plot
varImpPlot(rf_model, main = "Drivers of Coral Bleaching")
```
