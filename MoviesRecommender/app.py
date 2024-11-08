from flask import Flask, request, jsonify, render_template
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

app = Flask(__name__)

# Load your dataset
dataset = pd.read_csv('C:\\Users\\roopa\\OneDrive\\Desktop\\MoviesRecommender\\movies.csv')

# Fill missing values and create combined features
dataset['genre'] = dataset['genre'].fillna('unknown')
dataset['overview'] = dataset['overview'].fillna('')
dataset['combined_features'] = dataset['genre'] + ' ' + dataset['overview']

# Create TF-IDF Matrix
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
tfidf_matrix = tfidf.fit_transform(dataset['combined_features'])

# Calculate Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Define your OMDb API key here
OMDB_API_KEY = "6f5cb803"

def recommend_movies_hierarchical(title_substring, num_recommendations=10):
    title_substring = title_substring.lower().strip()
    filtered_titles = dataset[dataset['title'].str.lower().str.contains(title_substring)]

    if filtered_titles.empty:
        words = title_substring.split()
        if len(words) > 1:
            substring_without_last = ' '.join(words[:-1])
            filtered_titles = dataset[dataset['title'].str.lower().str.contains(substring_without_last)]

    if filtered_titles.empty:
        return []

    filtered_indices = filtered_titles.index.tolist()
    filtered_cosine_sim = cosine_sim[filtered_indices, :][:, filtered_indices]
    sim_scores = list(enumerate(filtered_cosine_sim.sum(axis=1)))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [filtered_indices[i[0]] for i in sim_scores[:num_recommendations]]

    recommendations = []
    for idx in movie_indices:
        movie_title = dataset['title'].iloc[idx]
        release_date = dataset['release_date'].iloc[idx]  # Get the full release date

        # Make the API request with the correct URL format
        response = requests.get(f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_API_KEY}")
        movie_data = response.json()

        # Log the response for debugging
        logging.debug(f"OMDb API response for '{movie_title}': {movie_data}")

        if movie_data.get("Response") == "True":
            recommendations.append({
                'title': movie_title,
                'overview': movie_data.get("Plot", "Overview not available"),
                'poster_url': movie_data.get("Poster", "https://via.placeholder.com/300x450?text=No+Image+Available"),
                'rating': movie_data.get("imdbRating", "N/A"),
                'popularity': movie_data.get("Metascore", "N/A"),
                'release_date': release_date  # Store the full release date
            })
        else:
            logging.warning(f"Could not fetch data for '{movie_title}': {movie_data.get('Error')}")

    return recommendations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title', '')
    if not title:
        return jsonify({"error": "Please provide a movie title."}), 400

    recommendations = recommend_movies_hierarchical(title)
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True)
