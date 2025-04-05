import pandas as pd
from urllib.parse import urlparse
import difflib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

known_legit_domains = [
    "google.com", "facebook.com", "amazon.com", "apple.com", "microsoft.com",
    "paypal.com", "youtube.com", "instagram.com", "linkedin.com", "twitter.com",
    "snapchat.com", "whatsapp.com", "netflix.com", "chatgpt.com", "openai.com"
]

def is_similar_to_known(domain, known_list, threshold=0.75):
    for legit in known_list:
        sim = difflib.SequenceMatcher(None, domain, legit).ratio()
        if sim > threshold and domain != legit:
            return True
    return False

def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc.lower().split(":")[0]

    return {
        "url_length": len(url),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(url.count(c) for c in ['@', '-', '_', '%', '/']),
        "num_subdomains": parsed.netloc.count('.') - 1,
        "uses_https": int(parsed.scheme == "https"),
        "has_phishing_keyword": int(any(k in url.lower() for k in ['login', 'verify', 'secure', 'account', 'update'])),
        "has_risky_tld": int(any(url.endswith(tld) for tld in [".xyz", ".tk", ".ml", ".gq", ".cf"])),
        "typo_suspected": int(is_similar_to_known(domain, known_legit_domains)),
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

df = pd.read_csv(r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model3\updated_phishing_dataset.csv")
X = df.drop(columns=["is_phishing"])
y = df["is_phishing"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

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

print("\n🌍 URL Classification Results:")
for url, result in zip(urls, predictions):
    print(f"{url} -> {'Phishing' if result == 1 else 'Legitimate'}")
