import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from feature_extraction import extract_features

# Load cleaned dataset
data=r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model2\cleaned_dataset3.csv"
df = pd.read_csv(data)

# Split features and target
X = df.drop(columns=["label"])  # Features
y = df["label"]  # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

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
# Function to predict from a URL
def predict_url():
    url = input("\nEnter a URL: ")
    url_features = extract_features(url)

    # Define feature names exactly as in training
    feature_names = X.columns.tolist()  # Get column names from training data

    # Convert extracted features to a DataFrame
    url_features_df = pd.DataFrame(url_features, columns=feature_names)

    # Predict using the trained model
    prediction = model.predict(url_features_df)[0]

    # Display output in a clean format
    if prediction == 1:
        print("\n🚨 ALERT! This website is **FAKE**.\n🚫 Do NOT trust it. Stay safe!\n")
    else:
        print("\n✅ SAFE! This website appears to be **REAL** and trustworthy.\n")

# Take user input for prediction
predict_url()

