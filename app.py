from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
import pandas as pd
import os

app = Flask(__name__)

# ===================================================================
# -------------------- BIG MODEL DOWNLOAD HELPERS -------------------
# ===================================================================

def download_if_missing(url, path):
    """Download a file from `url` to `path` if it doesn't already exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading large model from: {url}")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def safe_load_pickle(path, name):
    """
    Safely load a pickle file.
    - Detects if the file is actually an HTML page (e.g. Google Drive warning).
    - Returns None instead of raising if anything goes wrong.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(1024)
            # Try to detect HTML instead of pickle
            try:
                text_head = head.decode(errors="ignore").lower()
            except Exception:
                text_head = ""

            if "<html" in text_head or "<!doctype html" in text_head or "google" in text_head:
                print(f"WARNING: {name} appears to be HTML (likely a web page), not a pickle file.")
                return None

            # Reset and load as pickle
            f.seek(0)
            return pickle.load(f)

    except Exception as e:
        print(f"WARNING: Failed to load {name} from {path}: {e}")
        return None


# ===================================================================
# --------------------- BIG MODEL URLS & PATHS ----------------------
# ===================================================================

# You can later replace these with Dropbox / GitHub Releases direct links.
SIMILARITY_URL = "https://drive.google.com/uc?export=download&id=1dzMH_3hp7-Y04hQfW8fJPo5CRqe8Nw8s"
COSINE_URL     = "https://drive.google.com/uc?export=download&id=12Crd2Mm1AmS7trMWcyGWxNWwNQnQJsaD"

SIMILARITY_PATH = "model/content_pkl/similarity.pkl"
COSINE_PATH     = "model/collab_pkl/cosine_sim.pkl"

# Download big models if missing
download_if_missing(SIMILARITY_URL, SIMILARITY_PATH)
download_if_missing(COSINE_URL,     COSINE_PATH)

# Try loading them safely (may be None if download failed / HTML returned)
similarity_content = safe_load_pickle(SIMILARITY_PATH, "similarity.pkl")
cosine_sim         = safe_load_pickle(COSINE_PATH, "cosine_sim.pkl")  # currently unused, but kept for future use


# ===================================================================
# ----------------------- CONTENT BASED DATA -------------------------
# ===================================================================

# Smaller files that SHOULD be shipped with your repo
movies_list = pickle.load(open('model/content_pkl/movie_list.pkl', 'rb'))


# ===================================================================
# -------------------- COLLABORATIVE FILTERING DATA ------------------
# ===================================================================

movies       = pd.read_csv('csv/movies.csv')
popular_df   = pickle.load(open('model/popular.pkl', 'rb'))
pt           = pickle.load(open('model/pt.pkl', 'rb'))
movies_cf    = pickle.load(open('model/movies.pkl', 'rb'))
similarity_cf = pickle.load(open('model/similarity_scores.pkl', 'rb'))


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
    """
    Recommend similar movies using the content-based similarity matrix.
    Gracefully returns [] if the big model isn't available.
    """
    if similarity_content is None:
        print("WARNING: similarity_content is None. Content-based recommendations are disabled.")
        return []

    if movie not in movies_list['title'].values:
        return []

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
    """
    Recommend similar movies using collaborative filtering.
    """
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
    # Basic home page that can link to both recommenders
    return render_template("index.html")


# ---------- CONTENT BASED ------------
@app.route("/content", methods=["GET", "POST"])
def content_page():
    movie_list_titles = movies_list['title'].values
    selected_movie = None
    recommendations = None

    if request.method == "POST":
        selected_movie = request.form.get("movie_name")
        recommendations = recommend_content(selected_movie)

    return render_template(
        "content_recommend.html",
        movie_list=movie_list_titles,
        selected_movie=selected_movie,
        recommendations=recommendations
    )


# ---------- COLLABORATIVE ------------
@app.route("/collab", methods=["GET", "POST"])
def collab_page():
    movie_list_titles = pt.index.values
    selected_movie = None
    recommendations = None

    if request.method == "POST":
        selected_movie = request.form.get("movie_name")
        recommendations = recommend_collab(selected_movie)

    return render_template(
        "collab_recommend.html",
        movie_list=movie_list_titles,
        selected_movie=selected_movie,
        recommendations=recommendations
    )


# ===================================================================
# ----------------------------- RUN APP ------------------------------
# ===================================================================

if __name__ == "__main__":
    # For local development only. On Render you use gunicorn.
    app.run(debug=True)
