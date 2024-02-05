import re
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

class CustomFlaskAppWithEncoder(Flask):
    json_provider_class = LazyJSONEncoder

app = CustomFlaskAppWithEncoder(__name__)

swagger_template = {
    'info': {
        'title': 'API Documentation for Data Processing and Modeling',
        'version': '1.0.0',
        'description': 'Dokumentasi API untuk Data Processing dan Modeling',
    },
    'host': LazyString(lambda: request.host),
}

swagger_config = {
    'headers': [],
    'specs': [
        {
            'endpoint': 'docs',
            'route': '/docs.json',
        }
    ],
    'static_url_path': '/flasgger_static',
    'swagger_ui': True,
    'specs_route': '/docs/',
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)
# Unduh kamus kata-kata stop words Bahasa Indonesia
nltk.download('stopwords')
nltk.download('punkt')  # Unduh data tokenisasi

# Inisialisasi set stop words
wordlist = set(stopwords.words('indonesian'))

# Membuat set kata-kata tambahan
additional_stopwords = set([
    "yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp',
    'biar', 'bikin', 'bilang', 'gak', 'ga', 'krn', 'nya', 'nih',
    'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 'jd', 'jgn',
    'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan',
    'loh', 'user', 'bukan', 'hanya', 'kata', 'gua', '&amp', 'yah'
    , 'orang', 'lu', 'url', 'gue', 'tp', 'gw', 'udah', 'lo', 'url'
    , 'dah', 'jg', 'org', 'emang', 'pake', 'no'
    , 'pa','ni','mah','iya','bgt','tu','gk','liat','mas','cak'
])

# Gabungkan set stop words NLTK dan set kata-kata tambahan
list_stopwords = wordlist.union(additional_stopwords)

# Function text cleaning with stopwords removal
def cleansing(sent):
    string = sent.lower()
    string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
    string = re.sub(r'https?://\S+|www\.\S+', '', string)
    string = re.sub(r'[^a-zA-Z0-9\s]', '', string)
    string = re.sub(r'x[0-9a-fA-F]+', '', string)
    string = re.sub(r'\s+', ' ', string).strip()
    tokens = word_tokenize(string)
    tokens = [word for word in tokens if word not in list_stopwords]
    return ' '.join(tokens)

# Upload model and vectorizer
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
model = pickle.load(open('model_classification.p', 'rb'))

# Endpoint for sentiment prediction
@app.route('/predict_sentiment', methods=['POST'])
@swag_from("docs/sentiment.yml", methods=['POST'])
def predict_sentiment():
    input_text = request.args.get('text')
    
    if not input_text:
        return jsonify({"error": "Input text is empty"}), 400

    text = tfidf_vectorizer.transform([cleansing(input_text)])

    # Predict using the model
    result = model.predict(text)[0]

    # Return a JSON response
    return jsonify({"prediction": str(result)})

# Endpoint for processing data from CSV
@app.route('/process_csv', methods=['POST'])
@swag_from('docs/nnfile.yml', methods=['POST'])
def predict_sentiment_csv():
    # Load the CSV dataset
    csv_path = request.files['file']

    try:
        dataset = pd.read_csv(csv_path, encoding='latin1')
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'CSV file is empty'}), 400
    except pd.errors.ParserError:
        return jsonify({'error': 'Error parsing CSV file. Please check the file format'}), 400

    # Get the column name from the request
    kolom = request.args.get('kolom')
    if kolom not in dataset.columns:
        return jsonify({'error': f"Column '{kolom}' not found in the dataset"}), 400

    # Assuming your dataset has a column named 'text' containing the text data
    dataset['cleaned_text'] = dataset[kolom].apply(cleansing)
    text_features = tfidf_vectorizer.transform(dataset['cleaned_text'])

    # Predict sentiment for each text in the dataset
    predictions = model.predict(text_features)

    # Save the predictions to a new column in the dataset
    dataset['predicted_sentiment'] = predictions

    # Create a new DataFrame with only the relevant columns
    result_df = dataset[['cleaned_text', 'predicted_sentiment']]

    # Convert the DataFrame to a list of dictionaries
    result_list = result_df.to_dict(orient='records')
    
    # Save the results to a CSV file
    output_csv_path = 'hasillabelingnn.csv'
    result_df.to_csv(output_csv_path, index=False)
    
    # return response    
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis Using Neural Network",
        'data': result_list
    }

    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run(debug=True)
