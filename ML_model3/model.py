import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

df = pd.read_csv(r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model3\synthetic_phishing_dataset.csv")

X = df.drop(columns=["URL", "is_phishing"])
y = df["is_phishing"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f}")

def predict_url(url):
    url_length = len(url)
    num_digits = sum(c.isdigit() for c in url)
    num_special_chars = len([c for c in url if not c.isalnum()])
    num_subdomains = url.count(".") - 1
    uses_https = 1 if url.startswith("https") else 0
    
    input_features = np.array([[url_length, num_digits, num_special_chars, num_subdomains, uses_https]])
    prediction = model.predict(input_features)[0]
    return "Phishing" if prediction == 1 else "Safe"

user_url = input("Enter a URL to check: ")
print(f"Prediction for {user_url}: {predict_url(user_url)}")
