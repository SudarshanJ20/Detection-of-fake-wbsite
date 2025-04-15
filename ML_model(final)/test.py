import pandas as pd

data = r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model(final)\phishing_cleaned.csv"
df = pd.read_csv(data)

print(df.head())
print(df.info())

