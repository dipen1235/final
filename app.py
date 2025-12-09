from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
import pandas as pd

app = Flask(__name__)

# ===================================================================
# ----------------------- CONTENT BASED DATA -------------------------
# ===================================================================

movies_list = pickle.load(open('model/content_pkl/movie_list.pkl', 'rb'))
similarity_content = pickle.load(open('model/content_pkl/similarity.pkl', 'rb'))

# ===================================================================
# -------------------- COLLABORATIVE FILTERING DATA ------------------
# ===================================================================

movies = pd.read_csv('csv/movies.csv')
popular_df = pickle.load(open('model/popular.pkl','rb'))
pt = pickle.load(open('model/pt.pkl','rb'))
movies_cf = pickle.load(open('model/movies.pkl','rb'))
similarity_cf = pickle.load(open('model/similarity_scores.pkl','rb'))

# ===================================================================
# ------------------------ HELPER FUNCTIONS --------------------------
# ===================================================================

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=c4e0cda33ee220d70e5882632cec6bc8&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    if poster_path:
        return "https://image.tmdb.org/t/p/original" + poster_path
    return "https://via.placeholder.com/300x450?text=No+Image"

# -------- CONTENT BASED ----------
def recommend_content(movie):
    index = movies_list[movies_list['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity_content[index])),
        key=lambda x: x[1],
        reverse=True
    )

    recommended = []
    for i in distances[0:5]:
        movie_id = movies_list.iloc[i[0]].movie_id
        recommended.append({
            "title": movies_list.iloc[i[0]].title,
            "movie_id": int(movie_id),
            "poster": fetch_poster(movie_id)
        })
    return recommended


# -------- COLLABORATIVE FILTERING ----------
def recommend_collab(movie_name):
    if movie_name not in pt.index:
        return []

    index = np.where(pt.index == movie_name)[0][0]
    similar_items = sorted(
        list(enumerate(similarity_cf[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommendations = []
    for i, score in similar_items:
        title_key = pt.index[i]
        temp_df = movies[movies['original_title'] == title_key]

        if temp_df.empty:
            continue

        temp_df = temp_df.drop_duplicates('original_title')
        title = temp_df['original_title'].values[0]
        movie_id = int(temp_df['movie_id'].values[0])

        recommendations.append({
            "title": title,
            "movie_id": movie_id,
            "poster": fetch_poster(movie_id)
        })

    return recommendations

# ===================================================================
# --------------------------- ROUTES --------------------------------
# ===================================================================

@app.route("/")
def home():
    return render_template("index.html")   # Show links to both recommenders


# ---------- CONTENT BASED ------------
@app.route("/content", methods=["GET", "POST"])
def content_page():
    movie_list = movies_list['title'].values
    selected_movie = None
    recommendations = None

    if request.method == "POST":
        selected_movie = request.form.get("movie_name")
        recommendations = recommend_content(selected_movie)

    return render_template(
        "content_recommend.html",
        movie_list=movie_list,
        selected_movie=selected_movie,
        recommendations=recommendations
    )


# ---------- COLLABORATIVE ------------
@app.route('/collab', methods=['GET', 'POST'])
def collab_page():
    movie_list = pt.index.values
    selected_movie = None
    recommendations = None

    if request.method == 'POST':
        selected_movie = request.form.get("movie_name")
        recommendations = recommend_collab(selected_movie)

    return render_template(
        "collab_recommend.html",
        movie_list=movie_list,
        selected_movie=selected_movie,
        recommendations=recommendations
    )


# ===================================================================
# ----------------------------- RUN APP ------------------------------
# ===================================================================

if __name__ == "__main__":
    app.run(debug=True)
