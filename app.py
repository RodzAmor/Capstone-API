from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
# import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, util
import torch


app = Flask(__name__)
CORS(app, supports_credentials=True)

# Load models outside
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
nlp = spacy.load("en_core_web_sm")

def load_data(file_path):
    return pd.read_csv(file_path)

def segment_text(text):
    # Splits by sentences
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def process_file(file_path, year, headline):
    data = pd.read_csv(file_path)
    data['Year'] = pd.to_datetime(data['Fill Date']).dt.year
    data_filtered = data[data['Year'] == int(year)]
    
    if data_filtered.empty:
        return None

    row = data_filtered.iloc[0]
    segments = segment_text(row['Risk Factors Text'])
    headline_embedding = model.encode([headline], batch_size=32, convert_to_tensor=True)

    # Encode all segments at once
    segment_embeddings = model.encode(segments, batch_size=32, convert_to_tensor=True)

    # Calculate similarities in a vectorized manner
    similarities = util.pytorch_cos_sim(headline_embedding, segment_embeddings)[0]

    highest_similarity, highest_similarity_idx = torch.max(similarities, dim=0)
    highest_similarity_segment = segments[highest_similarity_idx]

    return {
        "Ticker": row["Ticker"],
        "Company Name": row["Company Name"],
        "Fill Date": row["Fill Date"],
        "Year": row["Year"],
        "Headline": headline,
        "Highest Similarity Score": highest_similarity.item(),
        "Risk Section Representative Segment": highest_similarity_segment
    }


@app.route('/analyze', methods=['GET'])
def analyze():
    headline = request.args.get('headline', default=None, type=str)
    year = request.args.get('year', default=2024, type=str)
    print("Starting Analysis")
    print(f"Headline: {headline}")
    
    if headline is None:
        return jsonify({"Headline is required.": ""}), 500

    files = ['XOM.csv', 'ETR.csv', 'PXD.csv']
    results = []
    for file in files:
        result = process_file(file, year, headline)
        if result:
            results.append(result)

    results = pd.DataFrame(results)
    response = results.to_json(orient='records')

    return jsonify({"data": response})



@app.route('/example', methods=['GET'])
def example():
    result = load_data("example2.csv")
    result = result.to_json(orient='records')
    
    return jsonify({"data": result})

@app.route('/')
def index():
    return "API is Online!"

if __name__ == '__main__':
    app.run()

