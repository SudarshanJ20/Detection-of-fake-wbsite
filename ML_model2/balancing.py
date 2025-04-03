import pandas as pd
from imblearn.over_sampling import SMOTE

# Load the cleaned dataset
data=r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model2\cleaned_dataset.csv"
data_cleaned = pd.read_csv(data)

# Split features and target
X = data_cleaned.drop(columns=["label"])  # Replace with actual target column name
y = data_cleaned["label"]  # Replace with actual target column name

# Apply SMOTE
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Create a new balanced dataset
balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
balanced_data["label"] = y_resampled  # Add back the target column

# Save the balanced dataset
balanced_data.to_csv("balanced_dataset.csv", index=False)
print("Balanced dataset saved as 'balanced_dataset.csv'")
