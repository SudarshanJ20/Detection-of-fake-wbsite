import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv(r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model(final)\phishing_cleaned.csv")
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)

print("Performing 5-fold cross-validation...")
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Average CV Accuracy: {cv_scores.mean():.4f}\n")

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Model Evaluation on Test Data:")
print(f"Accuracy       : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision      : {precision_score(y_test, y_pred):.4f}")
print(f"Recall         : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score       : {f1_score(y_test, y_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

selected_features = ['HTTPS', 'AnchorURL', 'WebsiteTraffic', 'Index', 'SubDomains', 'PrefixSuffix-']

print("\nJust answer with: y = yes, n = no, s = suspicious / not sure\n")

def get_input(q, mapping):
    while True:
        ans = input(q + " (y/n/s): ").strip().lower()
        if ans in mapping:
            return mapping[ans]
        print("Please enter y, n, or s.")

answers = []

answers.append(get_input("1. Does the link start with 'https://'?", {'y': 1, 'n': -1, 's': -1}))
answers.append(get_input("2. Do most links/buttons take you to other websites?", {'y': -1, 's': 0, 'n': 1}))
answers.append(get_input("3. Is it a popular or well-known website?", {'y': 1, 'n': -1, 's': -1}))
answers.append(get_input("4. If you Google the site, does it show up?", {'y': 1, 'n': -1, 's': -1}))
answers.append(get_input("5. Does the site name have lots of dots (e.g. a.b.c.d.com)?", {'y': -1, 's': 0, 'n': 1}))
answers.append(get_input("6. Does the domain name have a dash (-)?", {'y': -1, 'n': 1, 's': -1}))

full_input = pd.DataFrame(columns=X.columns)
for col in full_input.columns:
    full_input[col] = [0]
for i, col in enumerate(selected_features):
    full_input[col] = answers[i]

prediction = clf.predict(full_input)

print("\nPrediction result:")
if prediction[0] == -1:
    print("This website is likely a PHISHING site.")
else:
    print("This website is LEGITIMATE.")
