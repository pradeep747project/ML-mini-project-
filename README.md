# app.py
from flask import Flask, render_template, request
import pickle
import os

# Load saved model and vectorizer (gracefully handle missing files)
model = None
vectorizer = None
MODEL_PATH = 'sentiment_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    try:
        model = pickle.load(open(MODEL_PATH, 'rb'))
        vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
    except Exception as e:
        # If loading fails, keep model as None and print the error for debugging
        print(f"Failed to load model/vectorizer: {e}")
else:
    print(f"Model files not found. Expected: {MODEL_PATH}, {VECTORIZER_PATH}")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    if model is None or vectorizer is None:
        # Inform the user to train/generate the model files
        return render_template('index.html', prediction_text=('Model not available. '
                                                             'Run `sentiment_model.py` to train and '
                                                             'create `sentiment_model.pkl` and `tfidf_vectorizer.pkl`.'))
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]

    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    return render_template('index.html', prediction_text=f'Sentiment: {sentiment}')

if __name__ == "__main__":
    app.run(debug=True)

    Flask>=2.0.0
scikit-learn>=1.0
pandas>=1.3
nltk>=3.6
numpy>=1.21
# sentiment_model.py
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Download stopwords if not already available
nltk.download('stopwords')
from nltk.corpus import stopwords

# 1️⃣ Load Dataset (You can replace this with any CSV from Kaggle)
# Example: "twitter_sentiments.csv" with columns: ['text', 'sentiment']
data = pd.read_csv("https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv")

# 2️⃣ Clean Text
def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # remove mentions
    text = re.sub(r'#', '', text)               # remove hashtags
    text = re.sub(r'RT[\s]+', '', text)        # remove retweets
    text = re.sub(r'https?:\/\/\S+', '', text)  # remove URLs
    text = text.lower()
    return text

data['clean_text'] = data['tweet'].apply(clean_text)

# 3️⃣ Split Data
X = data['clean_text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Convert Text to Vectors (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000, stop_words=stopwords.words('english'))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5️⃣ Train Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 6️⃣ Evaluate
y_pred = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# 7️⃣ Save Model & Vectorizer
pickle.dump(model, open('sentiment_model.pkl', 'wb'))
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

print("✅ Model and Vectorizer saved successfully!")
