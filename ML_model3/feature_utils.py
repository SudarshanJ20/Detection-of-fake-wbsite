import re
import urllib.parse
import tldextract
import math

# List of common phishing-related keywords, suspicious TLDs, URL shorteners, and known typo patterns
phishing_keywords = ["login", "secure", "update", "verify", "account", "banking"]
risky_tlds = ["tk", "ml", "ga", "cf", "gq"]
shorteners = ["bit.ly", "goo.gl", "t.co", "tinyurl.com", "ow.ly", "is.gd", "buff.ly"]
suspicious_typos = {
    "paypal": ["paypa1", "paypol"],
    "google": ["g00gle", "goog1e"],
    "amazon": ["arnazon", "amaz0n"],
    "microsoft": ["micros0ft", "m1crosoft"],
}

# List of valid/common TLDs (this is a minimal set; you can extend it)
valid_tlds = {
    "com", "org", "net", "edu", "gov", "mil", "io", "co", "info", "biz", "us", "uk",
    "ca", "de", "fr", "jp", "ru", "in"
}

def extract_features(url):
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""

    ext = tldextract.extract(url)
    domain = ext.domain
    suffix = ext.suffix
    subdomain = ext.subdomain

    # Check if it's an IP address
    has_ip_address = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) else 0

    # Check for shortening service
    full_domain = f"{ext.domain}.{ext.suffix}"
    url_is_shortened = 1 if full_domain in shorteners else 0

    # Check for known phishing keyword
    has_keyword = 1 if any(k in url.lower() for k in phishing_keywords) else 0

    # Check for risky TLDs (from our custom list)
    risky_tld = 1 if suffix in risky_tlds else 0

    # Check for invalid TLD based on common valid TLDs
    invalid_tld = 1 if suffix not in valid_tlds else 0

    # Typo detection in known domains
    typo_suspected = 0
    for legit, typos in suspicious_typos.items():
        if any(t in hostname.lower() for t in typos):
            typo_suspected = 1
            break

    # URL entropy calculation
    def shannon_entropy(s):
        prob = [float(s.count(c)) / len(s) for c in set(s)]
        return -sum([p * math.log(p, 2) for p in prob])
    
    entropy = shannon_entropy(url) if url else 0

    return {
        "url_length": len(url),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": len(re.findall(r"[^\w]", url)),
        "num_subdomains": subdomain.count("."),
        "uses_https": 1 if parsed.scheme == "https" else 0,
        "has_phishing_keyword": has_keyword,
        "has_risky_tld": risky_tld,
        "invalid_tld": invalid_tld,  # new feature
        "typo_suspected": typo_suspected,
        "has_iframe": 0,
        "has_mouseover_script": 0,
        "redirect_count": 0,
        "domain_age_days": 1000,
        "page_rank": 0.5,
        "has_ssl_certificate": 1 if parsed.scheme == "https" else 0,
        "dns_record_present": 1,
        "whois_update_days": 300,
        "ip_location_risk": 0,
        "click_rate": 0.5,
        "bounce_rate": 0.3,
        "avg_time_on_page": 60,
        "has_ip_address": has_ip_address,
        "url_entropy": entropy,
        "url_has_port_number": 1 if parsed.port else 0,
        "url_is_shortened": url_is_shortened,
    }
