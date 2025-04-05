import pandas as pd
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import re
import whois
import socket
import dns.resolver
from datetime import datetime
from collections import Counter
import math

df = pd.read_csv(r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model3\phishing_dataset_with_new_features.csv")

X = df.drop(columns=["URL", "hosting_country", "is_phishing"])
y = df["is_phishing"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

def has_ip(url):
    return int(bool(re.match(r"https?://(\d{1,3}\.){3}\d{1,3}", url)))

def is_shortened(url):
    return int(any(x in url for x in ["bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly"]))

def calculate_entropy(url):
    p = [freq / len(url) for freq in Counter(url).values()]
    return round(-sum(pi * math.log2(pi) for pi in p), 3)

def check_dns(domain):
    try:
        dns.resolver.resolve(domain, 'A')
        return 1
    except:
        return 0

def get_whois_info(domain):
    try:
        w = whois.whois(domain)
        created = w.creation_date
        updated = w.updated_date

        if isinstance(created, list):
            created = created[0]
        if isinstance(updated, list):
            updated = updated[0]

        now = datetime.now()
        domain_age = (now - created).days if created else 0
        update_days = (now - updated).days if updated else 0
        return domain_age, update_days
    except:
        return 0, 0

typo_keywords = [
    'g00gle', 'faceb00k', 'amaz0n', 'appl3', 'micros0ft', 'paypa1', 'netf1ix',
    'youtub', 'instagrom', 'linkdin', 'twittter', 'snapchap', 'whatsap'
]

def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")
    
    domain_age, update_days = get_whois_info(domain)
    dns_present = check_dns(domain)
    entropy = calculate_entropy(url)

    return {
        "url_length": len(url),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(url.count(c) for c in ['@', '-', '_', '%', '/']),
        "num_subdomains": domain.count('.') - 1,
        "uses_https": int(parsed.scheme == "https"),
        "has_phishing_keyword": int(any(k in url.lower() for k in ['login', 'verify', 'secure', 'account', 'update'])),
        "has_risky_tld": int(any(url.endswith(tld) for tld in [".xyz", ".tk", ".ml", ".gq", ".cf"])),
        "typo_suspected": int(any(x in domain for x in typo_keywords)),
        "has_iframe": 0,
        "has_mouseover_script": 0,
        "redirect_count": 0,
        "domain_age_days": domain_age,
        "page_rank": 0.4,
        "has_ssl_certificate": int(parsed.scheme == "https"),
        "dns_record_present": dns_present,
        "whois_update_days": update_days,
        "ip_location_risk": 1,
        "click_rate": 0.4,
        "bounce_rate": 0.5,
        "avg_time_on_page": 60,
        "has_ip_address": has_ip(url),
        "url_entropy": entropy,
        "url_has_port_number": int(":" in parsed.netloc),
        "url_is_shortened": is_shortened(url)
    }

urls = []
print("Enter URLs (Type 'done' to finish):")
while True:
    url = input("Enter URL: ").strip()
    if url.lower() == "done":
        break
    urls.append(url)

features_list = [extract_features(url) for url in urls]
features_df = pd.DataFrame(features_list)
features_scaled = scaler.transform(features_df)

predictions = model.predict(features_scaled)

print("\n URL Classification Results ")
for url, result in zip(urls, predictions):
    print(f"{url} → {'Phishing' if result == 1 else 'Legitimate'}")

feature_names = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importance from Random Forest")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
