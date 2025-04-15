import pandas as pd

data = "code/phishing_cleaned.csv"
df = pd.read_csv(data)

print(df.head())
print(df.info())

