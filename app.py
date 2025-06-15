import pandas as pd
import numpy as np
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from flask import Flask, render_template, request

# Load data and preprocess (from movies.py)
movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')

movies = movies_df.merge(credits_df, on='title')
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.dropna(inplace=True)

def dect_to_list(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(dect_to_list)
movies['keywords'] = movies['keywords'].apply(dect_to_list)

def convert_cast(obj):
    L = []
    cnt = 0
    for i in ast.literal_eval(obj):
        while cnt < 3:
            L.append(i['name'])
            cnt += 1
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id','title','tags']]
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: " ".join(x))

ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

# TF-IDF Vectorization
# tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
# vectors = tfidf.fit_transform(new_df['tags']).toarray()

# Count Vectorization (commented out)
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

# Recommendation function (from movies.py, adapted for multiple inputs)
def recommend_multi(input_movies):
    all_recommendations = []
    for movie in input_movies:
        movie_index = new_df[new_df['title'] == movie].index
        if len(movie_index) == 0:
            print(f"Movie '{movie}' not found in the database.")
            continue
        movie_index = movie_index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
        for i in movies_list:
            all_recommendations.append((new_df.iloc[i[0]].title, i[1]))
    
    # Sort all recommendations by similarity score and take top 5
    all_recommendations.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = [movie for movie, _ in all_recommendations[:5]]
    
    return top_recommendations

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_movies = request.form.getlist('movies')
    user_movies = [movie.strip() for movie in user_movies if movie.strip()] # Clean and filter empty inputs
    if not user_movies:
        return "Please enter at least one movie.", 400
    
    recommendations = recommend_multi(user_movies)
    return render_template('recommendations.html', user_movies=user_movies, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)