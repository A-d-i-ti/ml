# Decision Tree Classification - General Template

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load your dataset
# -----------------------------
# Option 1: Load CSV file
data = pd.read_csv("your_dataset.csv")

# Option 2: Create data manually
# data = pd.DataFrame({
#     "Age": [22, 25, 47, 52, 46, 56],
#     "Salary": [20000, 25000, 50000, 60000, 52000, 70000],
#     "Purchased": ["No", "No", "Yes", "Yes", "Yes", "Yes"]
# })

# -----------------------------
# 2. Select input and output
# -----------------------------
# Change this according to your dataset
target_column = "Purchased"

X = data.drop(target_column, axis=1)
y = data[target_column]

# -----------------------------
# 3. Convert categorical data into numbers
# -----------------------------
X = pd.get_dummies(X)

# If target column is text, convert it into numbers
if y.dtype == "object":
    y = pd.factorize(y)[0]

# -----------------------------
# 4. Split data into training and testing
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 5. Create and train Decision Tree model
# -----------------------------
model = DecisionTreeClassifier(
    criterion="gini",      # or "entropy"
    max_depth=3,           # change or remove if needed
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 6. Make predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 7. Evaluate model
# -----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 8. Visualize Decision Tree
# -----------------------------
plt.figure(figsize=(15, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=True,
    filled=True
)
plt.show()

'''To edit it for your question, mainly change:
data = pd.read_csv("your_dataset.csv") , target_column = "Purchased"
For example, if your output column is Result, write:
target_column = "Result" '''
