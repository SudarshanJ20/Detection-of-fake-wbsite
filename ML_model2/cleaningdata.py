import pandas as pd
import numpy as np
import re

# Load the dataset
data=r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model2\dataset2.csv"
df = pd.read_csv(data)  # Ensure the dataset file is correct

# Data Cleaning
print("Before cleaning: ", df.shape)
df.drop_duplicates(inplace=True)  # Remove duplicate rows
df.dropna(inplace=True)  # Remove rows with missing values
print("After cleaning: ", df.shape)

# Selecting relevant features
columns_to_keep = [
    "URLLength", "DomainLength", "IsDomainIP", "NoOfSubDomain", "TLDLegitimateProb", "NoOfObfuscatedChar",
    "LetterRatioInURL", "DegitRatioInURL", "IsHTTPS", "NoOfURLRedirect", "NoOfSelfRedirect", "HasExternalFormSubmit",
    "HasHiddenFields", "HasPasswordField", "HasSocialNet", "URLSimilarityIndex", "CharContinuationRate",
    "HasObfuscation", "NoOfExternalRef", "label"
]
df = df[columns_to_keep]

# Balance the dataset (if needed)
class_counts = df["label"].value_counts()
print("Class distribution before balancing:\n", class_counts)
if abs(class_counts[0] - class_counts[1]) > 0.1 * len(df):
    min_class = class_counts.idxmin()
    df_balanced = df.groupby("label").apply(lambda x: x.sample(class_counts.min(), random_state=42)).reset_index(drop=True)
    print("Dataset balanced.")
else:
    df_balanced = df

df = df_balanced

# Save cleaned dataset
df.to_csv("cleaned_dataset3.csv", index=False)
print("Cleaned dataset saved as cleaned_dataset.csv")
