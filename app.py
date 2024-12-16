from flask import Flask, request, render_template
import joblib
import re
import requests
from bs4 import BeautifulSoup
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

app = Flask(__name__)

# Path ke data TF-IDF dan model (disesuaikan dengan struktur folder Anda)
tfidf_data_path = os.path.join(app.root_path, 'model', 'Data_TFIDF.pkl')
#tfidf_data_path = os.path.join(app.root_path, 'model', 'Data_TFIDF_v3.pkl')
model_path = os.path.join(app.root_path, 'model', 'SVC.pkl')

# Muat data TF-IDF dan model
tfidf, features, labels = joblib.load(tfidf_data_path)
model = joblib.load(model_path)

# Preprocessing function
def regeks(teks):
    if isinstance(teks, str):
        teks = teks.lower()
        teks = re.sub(r'<[^>]*>', '', teks)
        teks = re.sub(r'http?://\S+ ?', '', teks)
        teks = re.sub(r'[^\w\s]', '', teks)
        teks = re.sub(r'[0-9]+', ' ', teks)
        teks = re.sub(r'[^\x00-\x7f]', '', teks)
        stop_words = stopwords.words('english') + \
                     stopwords.words('french') + \
                     stopwords.words('russian') + \
                     stopwords.words('indonesian')
        teks = ' '.join([word for word in teks.split() if word not in stop_words])
        lemmatizer = WordNetLemmatizer()
        teks = ' '.join([lemmatizer.lemmatize(word) for word in teks.split()])
        return teks
    return teks

# Function to scrape content from URL
def scrape_url(url):
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url  # Add https:// if protocol is not specified

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'}
    for attempt in range(2):  # Try up to 2 times
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Ensure request was successful
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} for {url} failed with error: {e}. Retrying...")
            time.sleep(2)  # Wait 2 seconds before retrying
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    content = scrape_url(url)

    if content is None:
        return render_template('index.html', error=f"Failed to fetch content from {url}")
    else:
        soup = BeautifulSoup(content, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        text_content = soup.get_text(separator=' ')
        processed_text = regeks(text_content)
        new_text_features = tfidf.transform([processed_text]).toarray()
        prediction = model.predict(new_text_features)
        predicted_category = prediction[0]
        return render_template('index.html', prediction=predicted_category, url=url)

if __name__ == '__main__':
    app.run(debug=True)