import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

valid_labels = [
    "grocery",
    "produce",
    "dairy",
    "meat",
    "seafood",
    "deli",
    "bakery",
    "liquor",
    "frozen",
    "canned",
    "beverages",
    # "sauces",
    # "nuts",
    "home",
    "health",
    # "non-ingredient",
    # "uncategorized",
]

def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('____', 1)
            if len(parts) == 2:
                category, text = parts
                if category in valid_labels:
                    data.append(text)
                    labels.append(category)
    return data, labels

file_path = 'data.txt'
data, labels = load_data(file_path)

print(f"Total data points: {len(data)}")
label_counts = pd.Series(labels).value_counts()
print("Category distribution:\n", label_counts)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

model_path = 'ingredient_classifier.joblib'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

def classify_ingredient(ingredient_text):
    return model.predict([ingredient_text])[0]

ingredient = "1 pound of frozen chicken breast"
category = classify_ingredient(ingredient)
print(f"The category for '{ingredient}' is '{category}'.")

