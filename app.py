from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util
import torch
import yfinance as yf

app = Flask(__name__)
# cors = CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "methods": "*"}}, supports_credentials=True)
# cors = CORS(app, resources={r"/*": {"origins": "*"}})
cors = CORS(app, supports_credentials=True)




model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
nlp = spacy.load("en_core_web_sm")

def load_data(file_path):
    return pd.read_csv(file_path)

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

# Will need to discuss this on the document as a concern
# Problems with text too long
# def truncate_text(text, max_length=50000):
#     return text if len(text) <= max_length else text[:max_length]

def process_file(file_path, year, headline):
    data = pd.read_csv(file_path)
    data['Year'] = pd.to_datetime(data['Fill Date'], utc=True).dt.year
    # data['Year'] = pd.to_datetime(data['Fill Date']).dt.year
    data_filtered = data[data['Year'] == int(year)]
    
    if data_filtered.empty:
        return None

    row = data_filtered.iloc[0]
    segments = segment_text(row['Risk Factors Text'])
    headline_embedding = model.encode([headline], batch_size=32, convert_to_tensor=True)

    segment_embeddings = model.encode(segments, batch_size=32, convert_to_tensor=True)

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

    file = f"./companies/{ticker.upper()}.csv"
    result = [process_file(file, year, headline)]

    result = pd.DataFrame(result)
    result = result.to_json(orient='records')
    print("Finished Analysis\n")
    return result


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
    result = load_data("./companies/example.csv")
    result = result.to_json(orient='records')
    
    return jsonify({"data": result})

@app.route('/')
def index():
    return "API is Online!"

if __name__ == '__main__':
    app.run(debug=True)
