import pandas as pd
import re
from urllib.parse import urlparse

def extract_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path
    

    url_length = len(url)
    num_special_chars = len(re.findall(r'[-@_?=#/.]', url))
    num_digits = len(re.findall(r'\d', url))
    num_subdomains = domain.count('.')
    uses_https = 1 if parsed_url.scheme == 'https' else 0
    
    phishing_keywords = ['secure', 'login', 'verify', 'account', 'bank', 'free', 'update', 'win', 'click']
    contains_keyword = any(keyword in url.lower() for keyword in phishing_keywords)
    
    return [url_length, num_special_chars, num_digits, num_subdomains, uses_https, contains_keyword]

df = pd.read_csv(r"c:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model\dataset.csv")

df[['url_length', 'num_special_chars', 'num_digits', 'num_subdomains', 'uses_https', 'contains_keyword']] = df['URL'].apply(lambda x: pd.Series(extract_features(str(x))))

df.to_csv(r"c:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model\dataset_with_features.csv", index=False)


print("Feature extraction complete! New dataset saved as dataset_with_features.csv")
