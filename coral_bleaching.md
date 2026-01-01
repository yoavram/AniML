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

Result: The Recall for 'High' bleaching events has slightly improved from 32.8% (169 / (169 + 346)) to 33.8% (174 / (174 + 341)).

The Fix: Threshold Tuning
In standard classification, the model predicts "High" only if it is >50% sure. But for a marine biologist, missing a bleaching event is costly. You might want to flag a reef as "High Risk" even if the model is only 20% sure.
Prompt:
```
The class weighting didn't help enough. Let's try Threshold Moving.

Use the model to predict probabilities instead of classes (type='prob').

Create a histogram of the predicted probabilities for the 'High' class.

Create a new prediction: If the probability of 'High' is greater than 0.25 (instead of the default 0.5), classify it as 'High'.

Print the new Confusion Matrix. Did Recall improve?
```

What to expect: Recall for 'High' will jump up significantly (maybe to 70-80%).
False Positives will also jump (you will alarm for some healthy reefs).

- Original Model (No weighting): Recall for 'High' = 169 / (169 + 346) = 32.8%
- Weighted Model: Recall for 'High' = 174 / (174 + 341) = 33.8%
- Threshold Moving (0.25 threshold): Recall for 'High' = 268 / (268 + 247) = 52.0%

The Discussion: Ask the class: "As a park ranger, would you rather miss a dying reef (False Negative) or accidentally check a healthy reef (False Positive)?"

## Precision-Recall (PR) 
[PR Curve is significantly better than the standard ROC Curve](10.1371/journal.pone.0118432).

- ROC Curves can be misleading here because the "False Positive Rate" (x-axis) looks small just because you have so many "Low" samples (the denominator is huge).
- PR Curves focus purely on the minority class. They show the direct trade-off: "If I want to catch 80% of the bleaching (Recall), what will my False Alarm rate be (Precision)?"

Prompt:
```
Since the classes are imbalanced, accuracy is not a good metric. Use the ROCR package to calculate the performance measures. Plot the Precision-Recall Curve for the 'High' bleaching class. Calculate the Area Under the PR Curve (AUC-PR) to get a single score for model performance. Add a point on the plot showing where our current model (with default 0.5 threshold) sits.
```

Project the curve and ask the students to pick a point on the line:

The Conservative Scientist (Top-Left): High Precision, Low Recall.

"I only flag a reef if I am 100% sure. I will miss many bleaching events, but I will never cry wolf."

The Panic Button (Bottom-Right): Low Precision, High Recall.

"I flag everything that looks even slightly suspicious. I catch every bleaching event, but I waste money checking healthy reefs."

The Sweet Spot: The point closest to the top-right corner (1,1).

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
