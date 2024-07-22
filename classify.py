import joblib

def load_model(model_path):
    return joblib.load(model_path)

def classify_ingredient(model, ingredient_text):
    return model.predict([ingredient_text])[0]

model_path = 'ingredient_classifier.joblib'

model = load_model(model_path)

ingredient = "1 crate of zucchini"
category = classify_ingredient(model, ingredient)
print(f"The category for '{ingredient}' is '{category}'.")

