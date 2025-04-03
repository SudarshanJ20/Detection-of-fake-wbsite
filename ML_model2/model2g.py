import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from feature_extraction import extract_features

# Load dataset
data = r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model2\cleaned_dataset3.csv"
df = pd.read_csv(data)

# Split features and target
X = df.drop(columns=["label"])  # Features
y = df["label"]  # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost model (without 'use_label_encoder')
model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation Accuracy: {np.mean(cv_scores):.4f}")

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:\n", conf_matrix)

# Function to predict from a URL
def predict_url():
    url = input("Enter a URL: ")
    url_features = extract_features(url)
    prediction = model.predict(url_features)[0]

    if prediction == 1:
        print("\n WARNING: This website is **FAKE**. Do NOT proceed!\n")
    else:
        print("\n SAFE: This website appears to be **REAL** and trustworthy.\n")

# Take user input for prediction
predict_url()
