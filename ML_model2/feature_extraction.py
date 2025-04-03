import re
import numpy as np
import tldextract
from urllib.parse import urlparse

def extract_features(url):
    parsed_url = urlparse(url)
    extracted = tldextract.extract(url)

    features = {
        "URLLength": len(url),
        "DomainLength": len(extracted.domain),
        "IsDomainIP": 1 if re.match(r"^\d+\.\d+\.\d+\.\d+$", extracted.domain) else 0,
        "NoOfSubDomain": len(extracted.subdomain.split(".")) if extracted.subdomain else 0,
        "TLDLegitimateProb": 0.5,  # Placeholder (needs external source)
        "NoOfObfuscatedChar": sum(1 for c in url if c in ['%', '@']),
        "LetterRatioInURL": sum(c.isalpha() for c in url) / len(url),
        "DegitRatioInURL": sum(c.isdigit() for c in url) / len(url),
        "IsHTTPS": 1 if parsed_url.scheme == "https" else 0,
        "NoOfURLRedirect": url.count('//') - 1,  # Checking multiple redirects
        "NoOfSelfRedirect": 0,  # Placeholder (needs real request parsing)
        "HasExternalFormSubmit": 0,  # Placeholder (requires web scraping)
        "HasHiddenFields": 0,  # Placeholder (requires web scraping)
        "HasPasswordField": 0,  # Placeholder (requires web scraping)
        "HasSocialNet": 1 if any(social in url for social in ["facebook", "twitter", "linkedin", "instagram"]) else 0,
        "URLSimilarityIndex": 0.5,  # Placeholder (requires similarity checking)
        "CharContinuationRate": 0.5,  # Placeholder (needs NLP processing)
        "HasObfuscation": 1 if "%" in url or "@" in url else 0,
        "NoOfExternalRef": 0  # Placeholder (requires web page analysis)
    }

    return np.array(list(features.values())).reshape(1, -1)
