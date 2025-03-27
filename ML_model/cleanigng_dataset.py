import pandas as pd

data = r"c:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model\dataset.csv"

df = pd.read_csv(data)

missing_values = df.isnull().sum()

print("Missing Values in Each Column:")
print(missing_values[missing_values > 0])
