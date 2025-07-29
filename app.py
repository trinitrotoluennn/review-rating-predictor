from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Model ve vektörizer yükleniyor
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text")
    if not text:
        return render_template("index.html", prediction="No review was entered.")

    vect_text = vectorizer.transform([text])
    prediction = model.predict(vect_text)[0]
    return render_template("index.html", prediction=f"Predicted Rating: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
