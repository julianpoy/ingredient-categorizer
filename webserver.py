import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

model_path = 'ingredient_classifier.joblib'
model = joblib.load(model_path)

def classify_ingredient(model, ingredient_texts):
    return model.predict(ingredient_texts).tolist()

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        if not isinstance(data, list):
            return jsonify({'error': 'Invalid input, expected a list of strings'}), 400

        categories = classify_ingredient(model, data)

        return jsonify({'categories': categories})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)

