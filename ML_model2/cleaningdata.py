import pandas as pd
import numpy as np
import re

# Load the dataset
data = r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model2\dataset2.csv"  # Ensure the dataset file is correct
df = pd.read_csv(data)  # Ensure the dataset file is correct

# Data Cleaning
print("Before cleaning: ", df.shape)
df.drop_duplicates(inplace=True)  # Remove duplicate rows
df.dropna(inplace=True)  # Remove rows with missing values
print("After cleaning: ", df.shape)

def extract_features(url):
    """Extract numerical features from a URL."""
    return [
        len(url),  # URL Length
        url.count('.'),  # Number of dots
        1 if url.startswith("https") else 0,  # HTTPS presence
        any(word in url.lower() for word in ["login", "secure", "bank", "verify", "account", "update", "free"]),  # Suspicious words
        len(re.findall(r"[?&=%]", url))  # Number of special characters
    ]

# Apply feature extraction
df_features = df["URL"].apply(extract_features).apply(pd.Series)
df_features.columns = ["url_length", "num_dots", "https", "suspicious_words", "special_chars"]

df["label"] = df["label"].astype(int)  # Ensure labels are integers

df_final = pd.concat([df_features, df["label"]], axis=1)  # Keep only features + labels

# Balance the dataset (if needed)
class_counts = df_final["label"].value_counts()
print("Class distribution before balancing:\n", class_counts)
if abs(class_counts[0] - class_counts[1]) > 0.1 * len(df_final):
    min_class = class_counts.idxmin()
    df_balanced = df_final.groupby("label").apply(lambda x: x.sample(class_counts.min(), random_state=42)).reset_index(drop=True)
    print("Dataset balanced.")
else:
    df_balanced = df_final

df_final = df_balanced

# Save cleaned dataset
df_final.to_csv("cleaned_dataset.csv", index=False)
print("Cleaned dataset saved as cleaned_dataset.csv")
