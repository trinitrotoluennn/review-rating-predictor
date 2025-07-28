import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Model ve vektörleştiriciyi yükle
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "Metin girilmedi."}), 400

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]

    return jsonify({"score": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
