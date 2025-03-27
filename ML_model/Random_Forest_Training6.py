import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load processed dataset
df = pd.read_csv(r"c:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model\dataset_with_features.csv")

# Check available columns
print("Dataset Columns:", df.columns)

# Drop non-numeric columns
non_numeric_columns = ['URL', 'Domain', 'TLD', 'Title']
df = df.drop(columns=[col for col in non_numeric_columns if col in df.columns])

# Define features and target variable
label_column = 'Label' if 'Label' in df.columns else df.columns[-1]  # Use last column if 'Label' is missing
X = df.drop(columns=[label_column])
Y = df[label_column]  # Target variable (0: Legit, 1: Phishing)

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10)
rf_model.fit(X_train, Y_train)

# Cross-validation to check for overfitting
cv_scores = cross_val_score(rf_model, X_train, Y_train, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")

# Predictions
Y_pred = rf_model.predict(X_test)

# Performance Evaluation
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
class_report = classification_report(Y_test, Y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

"""
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Phishing'], yticklabels=['Legit', 'Phishing'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
"""