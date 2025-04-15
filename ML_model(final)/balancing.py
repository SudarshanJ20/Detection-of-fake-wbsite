import pandas as pd

# Load the cleaned CSV
df = pd.read_csv("code/phishing_cleaned.csv")

# Check class distribution
print(df['class'].value_counts())
print(df['class'].value_counts(normalize=True))
