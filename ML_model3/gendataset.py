import pandas as pd
import numpy as np
import random
import string
import math

def generate_random_domain(phishing=False):
    brands = ["amazon", "google", "paypal", "microsoft", "apple", "facebook", "netflix"]
    tlds_safe = [".com", ".org", ".net", ".co"]
    tlds_risky = [".tk", ".ml", ".gq", ".xyz", ".cf"]
    
    brand = random.choice(brands)
    if phishing:
        typo = random.choice(["0", "1", "l", "z"])
        typo_brand = brand.replace("o", typo).replace("a", "@")
        sub = random.choice(["login", "secure", "verify", "update"])
        domain = f"http://{sub}-{typo_brand}{random.choice(tlds_risky)}"
    else:
        domain = f"https://www.{brand}{random.choice(tlds_safe)}"
    return domain

def calculate_entropy(s):
    prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
    return -sum([p * math.log(p) / math.log(2.0) for p in prob])

def has_ip_address(url):
    return int(bool(random.random() < 0.05 and "http://" in url and random.random() < 0.5))

def generate_features(url, phishing):
    parsed_url = url.lower()
    domain = parsed_url.split("//")[-1].split("/")[0]
    
    url_length = len(url)
    num_digits = sum(c.isdigit() for c in url)
    num_special_chars = sum(url.count(c) for c in ['@', '-', '_', '%', '/'])
    num_subdomains = domain.count('.') - 1
    uses_https = int(url.startswith("https"))
    has_phishing_keyword = int(any(k in url for k in ["login", "verify", "secure", "update", "account"]))
    has_risky_tld = int(any(url.endswith(tld) for tld in [".tk", ".ml", ".gq", ".xyz", ".cf"]))
    typo_suspected = int(any(typo in url for typo in ["amaz0n", "g00gle", "paypa1", "micros0ft", "faceb00k"]))
    has_iframe = int(phishing and random.random() < 0.6)
    has_mouseover_script = int(phishing and random.random() < 0.6)
    redirect_count = random.randint(0, 5 if phishing else 2)
    domain_age_days = random.randint(0, 300 if phishing else 4000)
    page_rank = round(random.uniform(0.0, 0.4 if phishing else 0.9), 2)
    has_ssl_certificate = int(uses_https and not phishing)
    dns_record_present = int(not phishing or random.random() > 0.1)
    whois_update_days = random.randint(0, 100 if phishing else 300)
    hosting_country = random.choice(["US", "RU", "CN", "DE", "FR", "IN"])
    ip_location_risk = int(hosting_country in ["RU", "CN"])
    click_rate = round(random.uniform(0.0, 0.5 if phishing else 0.9), 2)
    bounce_rate = round(random.uniform(0.4 if phishing else 0.1, 1.0), 2)
    avg_time_on_page = random.randint(0 if phishing else 20, 300)

    # New features
    entropy = round(calculate_entropy(url), 2)
    has_ip = has_ip_address(url)
    has_port = int(":" in domain and domain.split(":")[-1].isdigit())
    is_shortened = int(any(short in url for short in ["bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly"]))

    return [
        url, url_length, num_digits, num_special_chars, num_subdomains,
        uses_https, has_phishing_keyword, has_risky_tld, typo_suspected,
        has_iframe, has_mouseover_script, redirect_count,
        domain_age_days, page_rank, has_ssl_certificate,
        dns_record_present, whois_update_days, hosting_country,
        ip_location_risk, click_rate, bounce_rate, avg_time_on_page,
        has_ip, entropy, has_port, is_shortened, int(phishing)
    ]

# --- Generate Dataset ---

columns = [
    "URL", "url_length", "num_digits", "num_special_chars", "num_subdomains",
    "uses_https", "has_phishing_keyword", "has_risky_tld", "typo_suspected",
    "has_iframe", "has_mouseover_script", "redirect_count",
    "domain_age_days", "page_rank", "has_ssl_certificate",
    "dns_record_present", "whois_update_days", "hosting_country",
    "ip_location_risk", "click_rate", "bounce_rate", "avg_time_on_page",
    "has_ip_address", "url_entropy", "url_has_port_number", "url_is_shortened",
    "is_phishing"
]

data = []

for i in range(5000):
    data.append(generate_features(generate_random_domain(phishing=True), True))
    data.append(generate_features(generate_random_domain(phishing=False), False))

# Add 2% label noise
for i in random.sample(range(len(data)), int(0.02 * len(data))):
    data[i][-1] = 1 - data[i][-1]  # Flip label

# Convert and Save
df = pd.DataFrame(data, columns=columns)
df.to_csv("phishing_dataset_with_new_features.csv", index=False)
print("Dataset generated and saved as phishing_dataset_with_new_features.csv")
