from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch
import yfinance as yf
import os

app = Flask(__name__)
# cors = CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "methods": "*"}}, supports_credentials=True)
# cors = CORS(app, resources={r"/*": {"origins": "*"}})
cors = CORS(app, supports_credentials=True)


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
nlp = spacy.load("en_core_web_lg")

def load_data(file_path):
    return pd.read_csv(file_path)

def get_doc(text):
    doc = nlp(text)
    return doc

@app.route('/api/stock/<ticker>')
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    
    year = request.args.get('year', None)
    if not year:
        return jsonify({'error': 'Year parameter is required'}), 400

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    hist = stock.history(start=start_date, end=end_date)

    data = [{
        'date': index.strftime('%Y-%m-%d'),
        'close': row['Close']
    } for index, row in hist.iterrows()]

    return jsonify(data)

def segment_text(text):
    # Splits by sentences
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def get_filtered_data(file_path, year):
    data = pd.read_csv(file_path)
    data['Year'] = pd.to_datetime(data['Fill Date'], utc=True).dt.year
    data_filtered = data[data['Year'] == int(year)]
    return data_filtered

def get_dict(row, headline, score, segment):
    return {
        "Ticker": row["Ticker"],
        "Company Name": row["Company Name"],
        "Fill Date": row["Fill Date"],
        "Year": row["Year"],
        "Headline": headline,
        "Highest Similarity Score": score,
        "Risk Section Representative Segment": segment
    }

# Will need to discuss this on the document as a concern
# Problems with text too long
# def truncate_text(text, max_length=50000):
#     return text if len(text) <= max_length else text[:max_length]

def process_file(file_path, year, headline):
    data_filtered = get_filtered_data(file_path, year)
    if data_filtered.empty:
        return None
    row = data_filtered.iloc[0]
    segments = segment_text(row['Risk Factors Text'])
    headline_embedding = model.encode([headline], batch_size=32, convert_to_tensor=True)
    segment_embeddings = model.encode(segments, batch_size=32, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(headline_embedding, segment_embeddings)[0]
    highest_similarity, highest_similarity_idx = torch.max(similarities, dim=0)
    highest_similarity_segment = segments[highest_similarity_idx]
    return get_dict(row, headline, highest_similarity.item(), highest_similarity_segment)
    
def calculate_nlp_similarity(file_path, year, headline):
    data_filtered = get_filtered_data(file_path, year)
    if data_filtered.empty:
        return None
    similarities = []
    row = data_filtered.iloc[0]
    segment_doc = get_doc(row['Risk Factors Text'])
    headline_doc = get_doc(headline)
    for sent in segment_doc.sents:
        similarity_score = headline_doc.similarity(sent)
        similarities.append((sent.text, similarity_score))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return get_dict(row, headline, similarities[0][1], similarities[0][0])
    
def calculate_bert_similarity(file_path, year, headline):
    data_filtered = get_filtered_data(file_path, year)
    if data_filtered.empty:
        return None
    similarities = []
    row = data_filtered.iloc[0]
    segments = segment_text(row['Risk Factors Text'])
    for segment in segments:
        input_ids = tokenizer.encode(headline, segment, return_tensors="pt", max_length=5120, truncation=True)
        with torch.no_grad():
            outputs = model(input_ids)
            embeddings = outputs[0].squeeze(0)
            similarity = torch.cosine_similarity(embeddings[0], embeddings[1], dim=0)
        similarities.append((segment, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return get_dict(row, headline, similarities[0][1], similarities[0][0])
    
def calculate_tfidf_similarity(file_path, year, headline):
    data_filtered = get_filtered_data(file_path, year)
    if data_filtered.empty:
        return None
    similarities = []
    row = data_filtered.iloc[0]
    segments = segment_text(row['Risk Factors Text'])
    tfidf_vectorizer = TfidfVectorizer()
    for segment in segments:
        tfidf_matrix = tfidf_vectorizer.fit_transform([headline, segment])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        similarities.append((segment, similarity_matrix[0][1]))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return get_dict(row, headline, similarities[0][1], similarities[0][0])
    
def get_jaccard_similarity(sentence1, sentence2):
    set1 = set(sentence1.split())
    set2 = set(sentence2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    jaccard_similarity = intersection / union if union != 0 else 0  # Avoid division by zero
    return jaccard_similarity

def calculate_jaccard_similarity(file_path, year, headline):
    data_filtered = get_filtered_data(file_path, year)
    if data_filtered.empty:
        return None
    similarities = []
    row = data_filtered.iloc[0]
    segments = segment_text(row['Risk Factors Text'])
    for segment in segments:
        similarities.append((segment, get_jaccard_similarity(headline, segment)))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return get_dict(row, headline, similarities[0][1], similarities[0][0])

def find_csv_directory(ticker):
    for root, dirs, files in os.walk('./companies'):
        for file in files:
            if file == f"{ticker.upper()}.csv":
                return root
    return None 

@app.route('/analyze-company', methods=['GET'])
def analyze_company():
    headline = request.args.get('headline', default=None, type=str)
    year = request.args.get('year', default=2024, type=str)
    ticker = request.args.get('ticker', default=2024, type=str)
    print("Starting Analysis")
    print(f"Headline: {headline}")
    
    if headline is None:
        return jsonify({"Headline is required.": ""}), 500
    if year is None:
        return jsonify({"Year is required.": ""}), 500
    if year is None:
        return jsonify({"Ticker is required.": ""}), 500

    # file = f"./companies/{ticker.upper()}.csv"
    file = find_csv_directory(ticker) + f"/{ticker.upper()}.csv"
    if file is None:
        return None
    
    result = [process_file(file, year, headline)]

    result = pd.DataFrame(result)
    result = result.to_json(orient='records')
    print("Finished Analysis\n")
    return result

# Deprecated
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
        result = process_file("./companies/" + file, year, headline)
        if result:
            results.append(result)

    results = pd.DataFrame(results)
    response = results.to_json(orient='records')

    return response

@app.route('/company', methods=['GET'])
def company_risk_year():
    print("Company CSV requested")
    ticker = request.args.get('ticker', default=None, type=str)
    year = request.args.get('year', default=None, type=int)

    if ticker is None:
        return jsonify({"Ticker is required.": ""}), 500
    if year is None:    
        return jsonify({"Year is required.": ""}), 500

    file_name = f"./companies/{ticker.upper()}.csv"
    try:
        data = load_data(file_name)
        data['Year'] = pd.to_datetime(data['Fill Date']).dt.year
        data = data[data['Year'] == year]
        
        if data.empty:
            return jsonify({})

        return data.to_json(orient='records')

    except FileNotFoundError:
        print("File Not Found")
        return jsonify({"error": f"File for ticker {ticker.upper()} not found."}), 404
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route('/example', methods=['GET'])
def example():
    result = load_data("./example.csv")
    result = result.to_json(orient='records')
    
    return jsonify({"data": result})

@app.route('/get-tickers', methods=['GET'])
def extract_tickers():
    farma = request.args.get('farma', default=None, type=str)

    directory = "./companies"
    farma_maps = {
        "0": "",
        "1": "/Consumer/",
        "2": "/Manufacturing/",
        "3": "/HiTec/",
        "4": "/Health and Medical/",
        "5": "/Energy/",
        "6": "/Other including Finance/",
    }
    if farma:
        directory += farma_maps[farma]

    tickers = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                ticker = file.replace('.csv', '')
                tickers.append(ticker)

    print(tickers)
    return jsonify({"tickers": tickers})

@app.route('/get-files', methods=['GET'])
def get_files():
    farma = request.args.get('farma', default=None, type=str)
    search = request.args.get('search', default=None, type=str)
    if search:
        search = search.upper()

    
    directory = "./companies"


    farma_maps = {
        "1": ("/Consumer/", "Consumer"),
        "2": ("/Manufacturing/", "Manufacturing"),
        "3": ("/HiTec/", "HiTec"),
        "4": ("/Health and Medical/", "Health and Medical"),
        "5": ("/Energy/", "Energy"),
        "6": ("/Other including Finance/", "Other including Finance"),
    }

    tickers = []
    if farma and farma in farma_maps:
        directory += farma_maps[farma][0]
        sector_name = farma_maps[farma][1]
    else:
        sector_name = None


    for root, dirs, files in os.walk(directory):
        folder_name = os.path.relpath(root, "./companies").replace("/", "")
        for file in files:
            if file.endswith('.csv'):
                ticker = file.replace('.csv', '')
                if search != None and search not in ticker:
                    continue
                    
                if sector_name:
                    tickers.append((ticker, sector_name))
                else:
                    tickers.append((ticker, folder_name))

    return jsonify({"tickers": tickers})

@app.route('/download-csv/', methods=['GET'])
def download_csv():
    sector = request.args.get('sector', type=str)
    ticker = request.args.get('ticker', type=str)

    if not sector or not ticker:
        return "Please provide both sector and ticker parameters.", 400

    directory = "./companies"
    farma_maps = {
        "Consumer": "/Consumer/",
        "Manufacturing": "/Manufacturing/",
        "HiTec": "/HiTec/",
        "Health and Medical": "/Health and Medical/",
        "Energy": "/Energy/",
        "Other including Finance": "/Other including Finance/",
    }

    if sector in farma_maps:
        directory += farma_maps[sector]
    else:
        return "Invalid sector provided.", 400

    filename = f"{ticker}.csv"
    file_path = os.path.join(directory, filename)

    if not os.path.isfile(file_path):
        return f"File '{filename}' not found.", 404

    return send_file(file_path, as_attachment=True)

@app.route('/')
def index():
    return "API is Online!"

if __name__ == '__main__':
    app.run(debug=True)
