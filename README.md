# Review Rating Predictor

This project is a simple web app that predicts the star rating (1 to 5) of a product review using a trained machine learning model.

---

## 🔗 Live Demo
👉 [Click here to try it online](https://review-rating-predictor.onrender.com)

---

## 🚀 Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/trinitrotoluennn/review-rating-predictor.git
cd review-rating-predictor

2. Install Requirements
pip install -r requirements.txt
                  
3. Run the App
python app.py


🧠 Model Info
Data: English Amazon product reviews

Task: Predict review rating (1 to 5) from text

Vectorizer: TF-IDF

Classifier: Logistic Regression

Model and vectorizer are saved as model.pkl and vectorizer.pkl.

Project Structure
├── app.py              # Flask application
├── model.pkl           # Trained ML model
├── vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt    # Required libraries
├── templates/
│   └── index.html      # Web UI (HTML form)
└── README.md           # Project info

📬 Contact
Made with ❤️ by @trinitrotoluennn



