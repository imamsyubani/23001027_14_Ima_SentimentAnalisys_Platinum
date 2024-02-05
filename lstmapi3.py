import re, pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from flask import request
import flask
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class CustomFlaskAppWithEncoder(Flask):
    json_provider_class = LazyJSONEncoder

app = CustomFlaskAppWithEncoder(__name__)

swagger_template = dict(
    info = {
        'title' : LazyString(lambda: "API Documentation for Deep Learning"),
        'version' : LazyString(lambda: "1.0.0"),
        'description' : LazyString(lambda: "Dokumentasi API untuk  Deep Learning"),
    },
    host = LazyString(lambda: request.host)
)

swagger_config = {
    "headers" : [],
    "specs" : [
        {
            "endpoint": "docs",
            "route" : "/docs.json",
        }
    ],
    "static_url_path": "/flasgger_static",
    # "static_folder": "static",  # must be set by user
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template, config = swagger_config)

# Initialize variable
max_features = 100000
sentiment = ['negative','neutral','positive']

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


# Load feature extraction and model LSTM
file = open("x_pad_sequences.pickle",'rb')
feature_file_from_lstm = pickle.load(file)
file = open("tokenizer.pickle",'rb')
tokenizer_from_lstm = pickle.load(file)
file.close()

model_file_from_lstm = load_model('model fit lstmfix-1.h5')

# API Sentiment Analysis using LSTM
@swag_from("docs/sentimentlstm.yml", methods = ['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():

    #  input text and cleansing
    original_text = request.args.get('text')
    text = [cleansing(original_text)]

    # convert text ke vector
    feature = tokenizer_from_lstm.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])

    # predict sentiment
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    # return response    
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis Using LSTM",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        }
    }

    response_data = jsonify(json_response)
    return response_data

# Endpoint for processing data from CSV
@swag_from('docs/lstmfile.yml', methods=['POST'])
@app.route('/process_csv', methods=['POST'])
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
    text_features = tokenizer_from_lstm.texts_to_sequences(dataset['cleaned_text'])
    text_features = pad_sequences(text_features, maxlen=feature_file_from_lstm.shape[1])

    # Predict sentiment for each text in the dataset
    predictions = model_file_from_lstm.predict(text_features)
    get_sentiment = [sentiment[np.argmax(prediction)] for prediction in predictions]

    # simpan the predictions to a new column in the dataset
    dataset['predicted_sentiment'] = get_sentiment

    # Create a new DataFrame with only the relevant columns
    result_df = dataset[['cleaned_text', 'predicted_sentiment']]

    # Convert the DataFrame to a list of dictionaries
    result_list = result_df.to_dict(orient='records')
    
    # Save the results to a CSV file
    output_csv_path = 'hasillabelinglstm.csv'
    result_df.to_csv(output_csv_path, index=False)
    # return response
    
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis Using LSTM",
        'data': result_list
    }

    response_data = jsonify(json_response)
    return response_data


if __name__ == '__main__':
    app.run()

