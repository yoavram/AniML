# 🐧 Session 1: The Morphology Classifier
**Objective:** Can we train a Machine Learning model to identify penguin species (Adelie, Chinstrap, Gentoo) based only on their physical measurements?

**The Data:**
- **Source:** Palmer Station, Antarctica LTER.
- **Variables:** Bill length, bill depth, flipper length, body mass, sex, island.

**The Mission:**
Use the Gemini AI agent to build a **Random Forest Classifier**. We want to:
1. Load the data from URL.
2. Clean missing values.
3. Split into Training (80%) and Testing (20%) sets.
4. Train the model.
5. Evaluate it (How often is it right?).
6. Visualize what the model "learned" (Feature Importance).
7. Visualize the model decisions (Decision Boundary).

## Prompt
Type this into the Gemini chat pane or a comment in the code cell, then hit "Generate".
```
Write a Python script to load the penguins dataset from https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv. Drop rows with missing values. Convert the 'species' column to a numeric code if needed. Split the data into X (features) and y (target) and then into train/test sets (80/20). Train a Random Forest Classifier. Print the accuracy score and plot a confusion matrix using seaborn. Finally, plot a bar chart of the feature importances.
```

Bonus prompt - find two important features `feature1` and `feature2` and enter the prompt:
```
I want to visualize how the model makes decisions. Since we can't plot 4 dimensions, retrain a new Random Forest using only two features: 'feature1' and 'feature2'. Then, write code to plot the decision boundaries.
```

## Teaching Points

The "DropNA" Trap: Point out that `df.dropna()` removed rows. Ask: "Is this okay? What if we lost 50% of our data?" (In this dataset, we only lose a few rows, but it's a critical question).

Feature selection: Note that we only used the numeric columns (bill, flipper, etc.). Ask: "Would 'Island' be useful?" (Yes, because certain species only live on certain islands).

The Result: The accuracy should be very high (~97%).

### Feature Importance
The plot usually shows Flipper Length or Bill Dimensions as the top predictors. This aligns with biological taxonomy.

### Decision boundaries
Observation: Ask students to look at the boundaries. Are they smooth curves (like a circle) or jagged steps?

The Lesson: They will be jagged/blocky. This visually proves that Random Forests are built from Decision Trees (which make square cuts: x>5, y<3).

Comparison: If you have time, ask the Agent to "Do the same with Logistic Regression." The line will be perfectly straight. This shows the difference between linear and non-linear models instantly.

## Code

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Data
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df = pd.read_csv(url)

# 2. Clean Data (Drop NaNs)
df = df.dropna()

# 3. Prepare X (Features) and y (Target)
# We drop 'species' (target) and 'island'/'sex' (categorical) for simplicity, 
# or we can encoding them. For a simple demo, let's stick to numeric measurements.
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['species']

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. Evaluate
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# 8. Feature Importance
plt.figure(figsize=(8,4))
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', color='teal')
plt.title('What matters for Penguin ID?')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 1. Select just 2 features for visualization
X_vis = df[['bill_length_mm', 'flipper_length_mm']].values
y_vis = df['species'].astype('category').cat.codes # Ensure y is numeric

# 2. Retrain model on just these 2 features
rf_vis = RandomForestClassifier(n_estimators=100, random_state=42)
rf_vis.fit(X_vis, y_vis)

# 3. Define bounds for the meshgrid
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# 4. Predict for the whole grid
Z = rf_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 5. Plot
plt.figure(figsize=(10, 6))
# Create custom cmap: matches Seaborn/Matplotlib default colors often used
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, cmap=cmap_bold, edgecolor='k', s=20)

plt.xlabel('Bill Length (mm)')
plt.ylabel('Flipper Length (mm)')
plt.title('Random Forest Decision Boundaries (Penguins)')
plt.show()
```
