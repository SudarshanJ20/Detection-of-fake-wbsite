import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from feature_extraction import extract_features

# Load dataset
data_path = r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model2\cleaned_dataset3.csv"
df = pd.read_csv(data_path)

# Check class distribution
print("Class Distribution:")
print(df["label"].value_counts())
print("\n")

# Split features and target
X = df.drop(columns=["label"])
y = df["label"]

# Verify feature consistency
print("Features in Training:", list(X.columns))
sample_features = extract_features("http://test.com").shape
print("Features in Prediction Input Shape:", sample_features)
print("\n")

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation Precision:", precision_score(y_val, y_val_pred))
print("Validation Recall:", recall_score(y_val, y_val_pred))
print("Validation F1-score:", f1_score(y_val, y_val_pred))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))
