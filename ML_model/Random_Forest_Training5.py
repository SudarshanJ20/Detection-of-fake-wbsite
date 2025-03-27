import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.under_sampling import NearMiss

df = pd.read_csv(r"c:\\Users\\sudar\\Downloads\\Detection-of-fake-wbsite-\\ML_model\\dataset_with_features.csv")

#print("Dataset Columns:", df.columns)

non_numeric_columns = ['URL', 'Domain', 'TLD', 'Title']
df = df.drop(columns=[col for col in non_numeric_columns if col in df.columns])

label_column = 'Label' if 'Label' in df.columns else df.columns[-1]  
X = df.drop(columns=[label_column])
Y = df[label_column] 

nm = NearMiss(version=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
X_train_resampled, Y_train_resampled = nm.fit_resample(X_train, Y_train)

scaler = MinMaxScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

param_dist = {
    'n_estimators': [50, 100],  
    'max_depth': [10, 20, None],
    'class_weight': [None, {0: 1, 1: 4}],
    'min_samples_split': [5, 10]
}

rand_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist, 
                                 n_iter=5, cv=2, scoring='accuracy', n_jobs=-1, random_state=42)
rand_search.fit(X_train_resampled, Y_train_resampled)

rf_model = rand_search.best_estimator_
print(f"Best Parameters: {rand_search.best_params_}")

cv_scores = cross_val_score(rf_model, X_train_resampled, Y_train_resampled, cv=2, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")

Y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
class_report = classification_report(Y_test, Y_pred)

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