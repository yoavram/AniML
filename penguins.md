# 🐧 Session 1: The Morphology Classifier
**Objective:** Can we train a Machine Learning model to identify penguin species (Adelie, Chinstrap, Gentoo) based only on their physical measurements?

**The Data:**
- **URL**: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv
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

## Too easy?

The Palmer Penguins species are linearly separable.

- Biology: These three species are very distinct. Gentoos are huge (mass). Chinstraps have very different bills.
- The Geometry: Imagine the data points in space. You can literally draw a straight flat sheet (a plane) between the Adélies and the Gentoos.
- The Lesson: If Logistic Regression gets 100%, you don't need AI or Random Forests. Occam's Razor wins.

How to "Break" the Model (And prove RF is better)

To show the value of Random Forest, you need to make the problem harder.
The "Bill Only" Challenge: Flipper Length and Body Mass are dead giveaways. Let's see if the AI can identify them just by looking at their beaks.

The Prompt Adjustment:
```
Retrain the models using ONLY bill_length_mm and bill_depth_mm. Compare Random Forest vs Logistic Regression.
```
