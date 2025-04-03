import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def extract_features(url):
    """
    Placeholder function to extract features from a URL.
    Ensure that this function returns exactly the same features as used in training.
    """
    # Example: Replace with actual feature extraction logic
    features = {
        'feature_1': len(url),
        'feature_2': url.count('-'),
        'feature_3': url.count('.'),
        'feature_4': int(url.startswith('https')),
        'feature_5': int('login' in url),
        # Add all 56 features accordingly
    }
    return features

# Load dataset
df = pd.read_csv(r"c:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model\dataset_with_features.csv")

# Drop non-numeric columns
non_numeric_columns = ['URL', 'Domain', 'TLD', 'Title']
df = df.drop(columns=[col for col in non_numeric_columns if col in df.columns])

# Define features and labels
label_column = 'Label' if 'Label' in df.columns else df.columns[-1]
X = df.drop(columns=[label_column])
Y = df[label_column]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10)
rf_model.fit(X_train, Y_train)

def predict_url(url):
    """ Predict if a URL is fake or real. """
    features_dict = extract_features(url)
    feature_order = X.columns  # Ensure correct feature order
    
    # Convert extracted features into a DataFrame with correct column order
    features_df = pd.DataFrame([features_dict], columns=feature_order)
    
    # Standardize extracted features
    features_scaled = scaler.transform(features_df)
    
    # Make prediction
    prediction = rf_model.predict(features_scaled)
    result = "Fake" if prediction[0] == 1 else "Real"
    print(f"The website is: {result}")

# Take URL input from user
user_url = input("Enter a URL: ")
predict_url(user_url)
