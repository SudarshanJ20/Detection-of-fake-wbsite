import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# === Load your dataset ===
df = pd.read_csv(r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model3\phishing_website_dataset.csv")

# Drop non-numeric columns
df_model = df.drop(columns=["URL", "hosting_country"])
X = df_model.drop(columns=["is_phishing"])
y = df_model["is_phishing"]

# Fit the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# === Feature extraction ===
phishing_keywords = ['login', 'verify', 'secure', 'account', 'update']
risky_tlds = ['.xyz', '.tk', '.ml', '.gq']
typo_variants = ['g00gle', 'faceb00k', 'amaz0n', 'appl3', 'micros0ft']

def extract_features_from_url(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    full = domain + path

    # Basic URL structure features
    num_digits = sum(c.isdigit() for c in url)
    num_special = sum(url.count(c) for c in ['@', '-', '_', '%', '/'])
    subdomain_count = domain.count('.') - 1 if domain else 0
    uses_https = int(parsed.scheme == 'https')
    phishing_kw = int(any(k in url.lower() for k in phishing_keywords))
    risky_tld = int(any(url.lower().endswith(tld) for tld in risky_tlds))
    typo = int(any(typo in url.lower() for typo in typo_variants))

    # === Smart logic for placeholder values ===
    trusted_domains = [
        "google.com", "paypal.com", "amazon.com", "apple.com",
        "microsoft.com", "facebook.com", "linkedin.com", "dropbox.com",
        "netflix.com", "ebay.com", "icloud.com"
    ]

    is_trusted = any(trusted in domain.lower() for trusted in trusted_domains)

    if is_trusted:
        domain_age_days = 3000
        page_rank = 0.95
        has_ssl_certificate = 1
        dns_record_present = 1
        whois_update_days = 30
        ip_location_risk = 0
        click_rate = 0.85
        bounce_rate = 0.2
        avg_time_on_page = 220
    else:
        domain_age_days = 100
        page_rank = 0.4
        has_ssl_certificate = uses_https
        dns_record_present = 1
        whois_update_days = 180
        ip_location_risk = 1 if risky_tld else 0
        click_rate = 0.4
        bounce_rate = 0.5
        avg_time_on_page = 60

    # Placeholder (still static but less important for now)
    has_iframe = 0
    has_mouseover_script = 0
    redirect_count = 0

    # Final feature set
    feature_values = [
        len(url), num_digits, num_special, subdomain_count,
        uses_https, phishing_kw, risky_tld, typo,
        has_iframe, has_mouseover_script, redirect_count,
        domain_age_days, page_rank, has_ssl_certificate,
        dns_record_present, whois_update_days,
        ip_location_risk, click_rate, bounce_rate, avg_time_on_page
    ]

    feature_names = [
        "url_length", "num_digits", "num_special_chars", "num_subdomains",
        "uses_https", "has_phishing_keyword", "has_risky_tld", "typo_suspected",
        "has_iframe", "has_mouseover_script", "redirect_count",
        "domain_age_days", "page_rank", "has_ssl_certificate",
        "dns_record_present", "whois_update_days",
        "ip_location_risk", "click_rate", "bounce_rate", "avg_time_on_page"
    ]

    return pd.DataFrame([feature_values], columns=feature_names)


# === Input & Prediction ===
user_url = input("Enter a URL to check: ")
features_df = extract_features_from_url(user_url)
features_scaled = pd.DataFrame(scaler.transform(features_df), columns=features_df.columns)
prediction = rf_model.predict(features_scaled)[0]

# === Output ===
print("\n RESULT:", "Phishing" if prediction == 1 else "Legitimate")
