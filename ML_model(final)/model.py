# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ----- Step 1: Load the cleaned dataset -----
# Ensure "phishing_cleaned.csv" is in the "code/" folder
df = pd.read_csv("code/phishing_cleaned.csv")

# ----- (Optional) Quick class balance check -----
print("Class distribution:")
print(df['class'].value_counts(normalize=True))

# ----- Step 2: Split into features and labels -----
X = df.drop('class', axis=1)
y = df['class']

# ----- Step 3: Train-Test Split -----
# 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- Step 4: Train the Classifier -----
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# ----- Step 5: Make Predictions and Evaluate -----
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n--- Model Performance ---")
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n", report)
