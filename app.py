from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import pandas as pd
import re
import string
import os

app = Flask(__name__, template_folder=os.getcwd())  # Set the template folder to the current directory

# Set up path for images
app.config['IMAGE_FOLDER'] = 'images'

# Load models and vectorizer
LR_model = joblib.load('model/LR_model.pkl')
DT_model = joblib.load('model/DT_model.pkl')
GB_model = joblib.load('model/GB_model.pkl')
RF_model = joblib.load('model/RF_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def output_label(n):
    return "Not A Fake News" if n == 1 else "Fake News"

# Route to serve images from the 'images' folder
@app.route('/images/<filename>')
def send_image(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

# Route to serve styles.css and style.css from the current directory
@app.route('/styles.css')
def serve_styles():
    return send_from_directory(os.getcwd(), 'styles.css')

@app.route('/style.css')
def serve_style():
    return send_from_directory(os.getcwd(), 'style.css')

@app.route('/')
def home():
    return render_template('home.html')  # Home page template

@app.route('/verify')
def verify():
    return render_template('verify.html')  # Verify page template

@app.route('/login')
def login():
    return render_template('login.html')  # Login page template

@app.route('/verify-news')
def verify_news():
    return render_template('verify_news.html')

@app.route('/external-affairs')  # Fix the function name for the new route
def external_affairs():
    return render_template('external-affairs.html')  # Render the correct template

@app.route('/pib')
def pib():
    return render_template('pib.html')  # PIB page template

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    processed = wordopt(news)
    vect = vectorizer.transform([processed])

    pred_LR = LR_model.predict(vect)[0]
    pred_DT = DT_model.predict(vect)[0]
    pred_GB = GB_model.predict(vect)[0]
    pred_RF = RF_model.predict(vect)[0]

    return jsonify({
        'prediction_text_LR': output_label(pred_LR),
        'prediction_text_DT': output_label(pred_DT),
        'prediction_text_GB': output_label(pred_GB),
        'prediction_text_RF': output_label(pred_RF)
    })

if __name__ == "__main__":
    app.run(debug=True)
