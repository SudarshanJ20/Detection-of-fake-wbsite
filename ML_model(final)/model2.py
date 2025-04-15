import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load and train model
df = pd.read_csv("code/phishing_cleaned.csv")
X = df.drop('class', axis=1)
y = df['class']

clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# Feature names we're using
selected_features = ['HTTPS', 'AnchorURL', 'WebsiteTraffic', 'Index', 'SubDomains', 'PrefixSuffix-']

# Ask easy questions
print("Just answer with: y = yes, n = no, s = suspicious / not sure\n")

def get_input(q, mapping):
    while True:
        ans = input(q + " (y/n/s): ").strip().lower()
        if ans in mapping:
            return mapping[ans]
        print("Please enter y, n, or s.")

answers = []

# Q1
answers.append(get_input("1. Does the link start with 'https://'?", {'y': 1, 'n': -1, 's': -1}))

# Q2
answers.append(get_input("2. Do most links/buttons take you to other websites?", {'y': -1, 's': 0, 'n': 1}))

# Q3
answers.append(get_input("3. Is it a popular or well-known website?", {'y': 1, 'n': -1, 's': -1}))

# Q4
answers.append(get_input("4. If you Google the site, does it show up?", {'y': 1, 'n': -1, 's': -1}))

# Q5
answers.append(get_input("5. Does the site name have lots of dots (e.g. a.b.c.d.com)?", {'y': -1, 's': 0, 'n': 1}))

# Q6
answers.append(get_input("6. Does the domain name have a dash (-)?", {'y': -1, 'n': 1, 's': -1}))

# Build input DataFrame
full_input = pd.DataFrame(columns=X.columns)
for col in full_input.columns:
    full_input[col] = [0]
for i, col in enumerate(selected_features):
    full_input[col] = answers[i]

# Predict
prediction = clf.predict(full_input)

# Result
print("\n🔎 Prediction result:")
if prediction[0] == -1:
    print("This website is likely a PHISHING site. Stay safe!")
else:
    print("This website appears LEGITIMATE.")
