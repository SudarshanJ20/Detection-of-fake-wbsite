import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from feature_utils import extract_features

df = pd.read_csv(r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model3\updated_phishing_dataset.csv")

X = df.drop(columns=["is_phishing"])
y = df["is_phishing"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

print("\nTest URLs (type 'done' to stop):")
urls = []
while True:
    url = input("Enter URL: ").strip()
    if url.lower() == "done":
        break
    urls.append(url)

if urls:
    features = [extract_features(u) for u in urls]
    test_df = pd.DataFrame(features)
    test_df = test_df[X.columns]
    test_scaled = scaler.transform(test_df)
    predictions = model.predict(test_scaled)

    print("\n Prediction Results:")
    for u, p, feat in zip(urls, predictions, features):
        if feat["typo_suspected"] == 1:
            p = 1
        print(f"{u} -> {'Phishing' if p == 1 else 'Legitimate'}")
