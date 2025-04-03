import pandas as pd

# Load dataset (Update path if needed)
data_path = r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model2\cleaned_dataset3.csv"
df = pd.read_csv(data_path)

print("✅ Dataset loaded successfully!")
print("🔹 Initial Shape:", df.shape)

# Compute correlation of features with 'label'
correlations = df.corr()["label"].sort_values(ascending=False)

print("\n🔍 Feature Correlations with Label:")
print(correlations)

# Identify highly correlated features (Threshold > 0.95)
high_corr_features = correlations[correlations > 0.95].index.tolist()

# Remove 'label' from the list (since its correlation with itself is 1.0)
high_corr_features.remove("label")

if high_corr_features:
    print("\n⚠️ Highly correlated features detected:", high_corr_features)
    # Drop the highly correlated features
    df = df.drop(columns=high_corr_features)
    print("✅ Removed highly correlated features!")
else:
    print("\n✅ No highly correlated features found. Dataset is good!")

print("🔹 New Shape after feature removal:", df.shape)
