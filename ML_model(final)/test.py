import pandas as pd

data = "phishing_cleaned.csv"
df = pd.read_csv(data)

print(df.head())
print(df.info())

