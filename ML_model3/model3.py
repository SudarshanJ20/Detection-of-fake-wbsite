import pandas as pd
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model3\phishing_dataset_with_new_features.csv")
X = df.drop(columns=["URL", "hosting_country", "is_phishing"])
y = df["is_phishing"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def extract_features(url):
    parsed = urlparse(url)
    return {
        "url_length": len(url),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(url.count(c) for c in ['@', '-', '_', '%', '/']),
        "num_subdomains": parsed.netloc.count('.') - 1,
        "uses_https": int(parsed.scheme == "https"),
        "has_phishing_keyword": int(any(k in url.lower() for k in ['login', 'verify', 'secure', 'account', 'update'])),
        "has_risky_tld": int(any(url.endswith(tld) for tld in [".xyz", ".tk", ".ml", ".gq", ".cf"])),
        "typo_suspected": int(any(x in url.lower() for x in ['g00gle', 'faceb00k', 'amaz0n', 'appl3', 'micros0ft'])),
        "has_iframe": 0,
        "has_mouseover_script": 0,
        "redirect_count": 0,
        "domain_age_days": 100,
        "page_rank": 0.4,
        "has_ssl_certificate": int(parsed.scheme == "https"),
        "dns_record_present": 1,
        "whois_update_days": 180,
        "ip_location_risk": 1,
        "click_rate": 0.4,
        "bounce_rate": 0.5,
        "avg_time_on_page": 60,
        "has_ip_address": 0,
        "url_entropy": 3.5,
        "url_has_port_number": int(":" in parsed.netloc),
        "url_is_shortened": int(any(x in url for x in ["bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly"]))
    }

# === Take multiple URLs from the user ===
urls = []
print("Enter URLs (Type 'done' to finish):")
while True:
    url = input("Enter URL: ").strip()
    if url.lower() == "done":
        break
    urls.append(url)

# Extract features for all URLs
features_list = [extract_features(url) for url in urls]
features_df = pd.DataFrame(features_list)

# Scale the features
features_scaled = scaler.transform(features_df)

# Make predictions
predictions = model.predict(features_scaled)

# Print results
print("\n=== URL Classification Results ===")
for url, result in zip(urls, predictions):
    print(f"{url} → {'Phishing' if result == 1 else 'Legitimate'}")
