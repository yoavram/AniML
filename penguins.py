import matplotlib
matplotlib.use("Agg")  # non-interactive backend; no window needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df = pd.read_csv(url)

# Drop rows with missing values
df = df.dropna()

# Encode categorical features
df["species_code"] = df["species"].astype("category").cat.codes
df["island_code"] = df["island"].astype("category").cat.codes
df["sex_code"] = df["sex"].astype("category").cat.codes

feature_cols = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "island_code",
    "sex_code",
]

X = df[feature_cols]
y = df["species_code"]

# Keep species labels for display
species_labels = df["species"].astype("category").cat.categories.tolist()

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Accuracy
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=species_labels,
    yticklabels=species_labels,
    ax=axes[0],
)
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion Matrix")

# Feature importances
importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(
    ascending=False
)
sns.barplot(x=importances.values, y=importances.index, ax=axes[1], palette="viridis")
axes[1].set_title("Feature Importances")
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("penguins_rf_results.png", dpi=150)
print("Plot saved to penguins_rf_results.png")

# ── Decision boundary (2-feature model) ─────────────────────────────────────

two_features = ["bill_length_mm", "bill_depth_mm"]
X2 = df[two_features]
y2 = df["species_code"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf2.fit(X2_train, y2_train)

y2_pred = clf2.predict(X2_test)
print(f"Accuracy (2-feature model): {accuracy_score(y2_test, y2_pred):.4f}")

# Build a fine mesh over the feature space
pad = 0.5
x_min, x_max = X2["bill_length_mm"].min() - pad, X2["bill_length_mm"].max() + pad
y_min, y_max = X2["bill_depth_mm"].min() - pad, X2["bill_depth_mm"].max() + pad
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

Z = clf2.predict(
    pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=two_features)
).reshape(xx.shape)

# Colours: one per species (same palette for background and markers)
scatter_palette = ["#1f77b4", "#d62728", "#2ca02c"]

fig, ax = plt.subplots(figsize=(8, 6))

# Filled decision regions — same colors as markers, with transparency
for code, color in enumerate(scatter_palette):
    ax.contourf(xx, yy, Z, levels=[-0.5, code + 0.5],
                colors=[color], alpha=0.25)

# Scatter the actual data points (train = filled, test = outlined)
for code, (label, scolor) in enumerate(zip(species_labels, scatter_palette)):
    mask_train = (y2_train == code)
    mask_test  = (y2_test  == code)
    ax.scatter(
        X2_train.loc[mask_train, "bill_length_mm"],
        X2_train.loc[mask_train, "bill_depth_mm"],
        color=scolor, edgecolors="white", linewidths=0.5,
        s=60, label=f"{label} (train)", zorder=3,
    )
    ax.scatter(
        X2_test.loc[mask_test, "bill_length_mm"],
        X2_test.loc[mask_test, "bill_depth_mm"],
        color=scolor, edgecolors="black", linewidths=1.2,
        s=80, marker="^", label=f"{label} (test)", zorder=4,
    )

ax.set_xlabel("Bill Length (mm)")
ax.set_ylabel("Bill Depth (mm)")
ax.set_title("Random Forest Decision Boundaries\n(bill_length_mm vs bill_depth_mm)")
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig("penguins_decision_boundary.png", dpi=150)
print("Decision boundary plot saved to penguins_decision_boundary.png")
