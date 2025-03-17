import os
import json
import numpy as np
import pandas as pd
import torch
import torch_directml
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify

# ---------- Step 1: Load the Movie Dataset ----------
print("Loading movie dataset...")
movies_df = pd.read_csv('movies_metadata.csv')

# ---------- Step 2: Preprocess the Data ----------
def create_movie_text(row):
    title = row.get('title', '')
    genres = row.get('genres', '')
    overview = row.get('overview', '')
    tagline = row.get('tagline', '')
    original_language = row.get('original_language', '')
    popularity = row.get('popularity', '')
    # Include popularity in the combined text to influence the embeddings
    combined = f"{title} {genres} {overview} {tagline} {original_language} {popularity}"
    return combined.strip()

movies_df['combined_text'] = movies_df.apply(create_movie_text, axis=1)

# ---------- Step 3: Load the SentenceTransformer Model on DirectML Device ----------
print("Loading SentenceTransformer model on AMD GPU using DirectML...")
dml_device = torch_directml.device()  # Create DirectML device
model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')

# ---------- Step 4: Compute or Load Movie Embeddings using Option 1 (Reduced Batch Size) ----------
embeddings_file = 'movie_embeddings.npy'
movie_texts = movies_df['combined_text'].tolist()

if os.path.exists(embeddings_file):
    print("Loading precomputed embeddings...")
    # Load the numpy array and convert it to a torch tensor on the DirectML device
    movie_embeddings = torch.tensor(np.load(embeddings_file, allow_pickle=True))
else:
    print("Computing embeddings for each movie with reduced batch size (Option 1)...")
    movie_embeddings = model.encode(
        movie_texts,
        show_progress_bar=True,
        convert_to_tensor=True,
        batch_size=4  # Reduced batch size to lower memory usage
    )
    # Move tensor to CPU and convert to numpy array before saving
    np.save(embeddings_file, movie_embeddings.cpu().numpy())
    print("Embeddings saved for future use.")

# ---------- Step 5: Define the Recommendation Function ----------
def get_recommendations(query, top_n=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    # Compute cosine similarities and move to CPU for numpy conversion
    cosine_scores = util.cos_sim(query_embedding, movie_embeddings)[0].cpu().numpy()
    top_indices = np.argpartition(-cosine_scores, range(top_n))[:top_n]
    top_indices = top_indices[np.argsort(-cosine_scores[top_indices])]
    recommendations = movies_df.iloc[top_indices][
        ['id', 'title', 'genres', 'overview', 'release_date', 'poster_path', 'popularity']
    ].to_dict(orient='records')
    return recommendations

# ---------- Step 6: Host the API Server ----------
app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Please provide a query in JSON payload with key "query".'}), 400
    query = data['query']
    recommendations = get_recommendations(query)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
