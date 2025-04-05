import pandas as pd
from urllib.parse import urlparse
import difflib

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

input_path = r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model3\phishing_dataset_with_new_features.csv"
df = pd.read_csv(input_path)

features_list = []
for url in df["URL"]:
    features = extract_features(url)
    features_list.append(features)

df_features = pd.DataFrame(features_list)
df_features["is_phishing"] = df["is_phishing"]

output_path = r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model3\updated_phishing_dataset.csv"
df_features.to_csv(output_path, index=False)

print("Dataset generated and saved successfully.")
