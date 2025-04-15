import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
df = pd.read_csv(r"C:\Users\sudar\Downloads\Detection-of-fake-wbsite-\ML_model(final)\phishing_cleaned.csv")

# Features and target
X = df.drop('class', axis=1)
y = df['class']

# Check class distribution
print("Class distribution:")
print(y.value_counts())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize SVM with balanced class weight
clf = SVC(random_state=42, class_weight='balanced')

# Perform 5-fold cross-validation
print("\nPerforming 10-fold cross-validation...")
cv_scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Average CV Accuracy: {cv_scores.mean():.4f}\n")

# Train the model
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Evaluation
print("Model Evaluation on Test Data:")
print(f"Accuracy       : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision      : {precision_score(y_test, y_pred):.4f}")
print(f"Recall         : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score       : {f1_score(y_test, y_pred):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Legit", "Phishing"], yticklabels=["Legit", "Phishing"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (SVM)")
plt.tight_layout()
plt.show()

# Manual user input
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

# Create user input DataFrame
full_input = pd.DataFrame(columns=X.columns)
for col in full_input.columns:
    full_input[col] = [0]
for i, col in enumerate(selected_features):
    full_input[col] = answers[i]

# Scale user input
full_input_scaled = scaler.transform(full_input)

# Predict user input
prediction = clf.predict(full_input_scaled)

print("\nPrediction result:")
if prediction[0] == -1:
    print("⚠️ This website is likely a FAKE website.")
else:
    print("✅ This website is REAL.")
