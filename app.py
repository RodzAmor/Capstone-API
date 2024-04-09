from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
# import spacy
# from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app, supports_credentials=True)


def load_data(file_path):
    return pd.read_csv(file_path)

# def segment_text(text):
#   nlp = spacy.load("en_core_web_sm")
#   doc = nlp(text)
#   # Splits by sentences
#   return [sent.text.strip() for sent in doc.sents]
# def semantic_analysis(file_path, headline, model_name):
# data = load_data(file_path)
# model = SentenceTransformer(model_name)

# results = []

# for index, row in data.iterrows():
#     document = row["Risk Factors Text"]
#     segments = segment_text(document)  # Segment the document into sentences

#     # Initialize the result row for this document
#     result_row = {
#         "Ticker": row["Ticker"],
#         "Company Name": row["Company Name"],
#         "Fill Date": row["Fill Date"],
#         "Model Used": model_name
#     }

    
#     headline_embedding = model.encode(headline, convert_to_tensor=True)
#     highest_similarity = 0 
#     highest_similarity_segment = ""
    
#     for segment in segments:
#         segment_embedding = model.encode(segment, convert_to_tensor=True)
#         similarity = util.pytorch_cos_sim(headline_embedding, segment_embedding).item()

#         if similarity > highest_similarity:
#             highest_similarity = similarity
#             highest_similarity_segment = segment

    
#     if highest_similarity > 0.6:
#         evaluation = "High"
#     elif highest_similarity > 0.3:
#         evaluation = "Medium"
#     else:
#         evaluation = "Low"

#     result_row.update({
#         "Headline": headline,
#         "Highest Similarity Score": highest_similarity,
#         "Similarity Evaluation": evaluation,
#         "Risk Section Representative Segment": highest_similarity_segment
#     })

#     results.append(result_row)

# return pd.DataFrame(results)

@app.route('/analyze', methods=['GET'])
def analyze():
    # all_models = {
    #     "Small": "all-MiniLM-L6-v2",
    #     "Medium": "all-MiniLM-L12-v2",
    #     "Large": "all-roberta-large-v1",
    # }
    headline = request.args.get('headline', default=None, type=str)
    if headline is None:
        return jsonify({"Headline is required.": str(e)}), 500
    
    # similarity_df = semantic_analysis("XOM.csv", headline, "all-MiniLM-L6-v2")

    try:
        # df = semantic_analysis(file_path, headlines, model_name)
        # Convert DataFrame to JSON
        # result = similarity_df.to_json(orient='records')
        pass
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # return jsonify({"data": result})
    return "temp"


@app.route('/example', methods=['GET'])
def example():
    result = load_data("example.csv")
    result = result.to_json(orient='records')
    return jsonify({"data": result})

@app.route('/')
def index():
    return "API is Online!"

if __name__ == '__main__':
    app.run()

