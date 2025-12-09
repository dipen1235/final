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
    - Detects if the file is actually an HTML page (e.g. Drive/Dropbox warning).
    - Returns None instead of raising if anything goes wrong.
    """
    if not os.path.exists(path):
        print(f"WARNING: {name} file not found at {path}")
        return None

    try:
        with open(path, "rb") as f:
            head = f.read(2048)
            # Try to detect HTML instead of pickle
            try:
                text_head = head.decode(errors="ignore").lower()
            except Exception:
                text_head = ""

            if "<html" in text_head or "<!doctype html" in text_head:
                print(f"WARNING: {name} appears to be HTML (likely a web page), not a pickle file.")
                return None

            # Reset and load as pickle
            f.seek(0)
            obj = pickle.load(f)

            print(f"DEBUG: Loaded {name} successfully. type={type(obj)}")
            # If it's a numpy array or similar, try to print shape
            shape = getattr(obj, "shape", None)
            if shape is not None:
                print(f"DEBUG: {name} shape = {shape}")

            return obj

    except Exception as e:
        print(f"WARNING: Failed to load {name} from {path}: {e}")
        return None


# ===================================================================
# --------------------- BIG MODEL URLS & PATHS ----------------------
# ===================================================================

# Using Dropbox for similarity.pkl
SIMILARITY_URL = (
    "https://dl.dropboxusercontent.com/scl/fi/dm0ge53sgfo1kgrm1z73i/"
    "similarity.pkl?rlkey=gcc58x0oyuaitb3yhti3yknrb&st=l78hm9w1&dl=1"
)

# cosine_sim still from Google Drive (you can move this to Dropbox later)
COSINE_URL = "https://drive.google.com/uc?export=download&id=12Crd2Mm1AmS7trMWcyGWxNWwNQnQJsaD"

SIMILARITY_PATH = "model/content_pkl/similarity.pkl"
COSINE_PATH     = "model/collab_pkl/cosine_sim.pkl"

# Download big models if missing
download_if_missing(SIMILARITY_URL, SIMILARITY_PATH)
download_if_missing(COSINE_URL,     COSINE_PATH)

# Try loading them safely (may be None if download failed / HTML returned)
similarity_content = safe_load_pickle(SIMILARITY_PATH, "similarity.pkl")
cosine_sim         = safe_load_pickle(COSINE_PATH, "cosine_sim.pkl")  # currently unused


# ===================================================================
# ----------------------- CONTENT BASED DATA -------------------------
# ===================================================================

# Smaller files that SHOULD be shipped with your repo
movies_list = pickle.load(open('model/content_pkl/movie_list.pkl', 'rb'))
print(f"DEBUG: movies_list loaded, length={len(movies_list)}")


# ===================================================================
# -------------------- COLLABORATIVE FILTERING DATA ------------------
# ===================================================================

movies        = pd.read_csv('csv/movies.csv')
popular_df    = pickle.load(open('model/popular.pkl', 'rb'))
pt            = pickle.load(open('model/pt.pkl', 'rb'))
movies_cf     = pickle.load(open('model/movies.pkl', 'rb'))
similarity_cf = pickle.load(open('model/similarity_scores.pkl', 'rb'))

print(f"DEBUG: pt index length = {len(pt.index)}")
print(f"DEBUG: movies.csv rows = {len(movies)}")


# ===================================================================
# ------------------------ HELPER FUNCTIONS --------------------------
# ===================================================================

def fetch_movie_data(movie_id):
    """
    Fetch poster URL and structured movie details (for hover & stars & trailers).
    """
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=c4e0cda33ee220d70e5882632cec6bc8&language=en-US"
    data = requests.get(url).json()

    poster_path = data.get('poster_path')
    if poster_path:
        poster_url = "https://image.tmdb.org/t/p/original" + poster_path
    else:
        poster_url = "https://via.placeholder.com/300x450?text=No+Image"

    title = data.get("title") or data.get("original_title") or "Unknown title"
    release_date = data.get("release_date") or "Unknown date"
    year = release_date.split("-")[0] if release_date and release_date != "Unknown date" else "N/A"
    rating = data.get("vote_average") or 0.0
    rating_5 = round(float(rating) / 2.0, 1)  # convert 0–10 -> 0–5 scale

    overview = data.get("overview") or "No overview available."
    if len(overview) > 250:
        overview = overview[:250].rstrip() + "..."

    details = f"{title} ({year})\nRating: {rating_5}/5\n\n{overview}"

    return {
        "poster_url": poster_url,
        "title": title,
        "year": year,
        "rating_10": float(rating),
        "rating_5": rating_5,
        "overview": overview,
        "details": details,
    }


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
        print(f"WARNING: '{movie}' not found in movies_list titles.")
        return []

    index = movies_list[movies_list['title'] == movie].index[0]

    try:
        row = similarity_content[index]
    except Exception as e:
        print(f"ERROR: Failed to index similarity_content with index {index}: {e}")
        return []

    distances = sorted(
        list(enumerate(row)),
        key=lambda x: x[1],
        reverse=True
    )

    recommended = []
    # skip [0] because it's the same movie
    for i in distances[1:6]:
        movie_id = int(movies_list.iloc[i[0]].movie_id)
        meta = fetch_movie_data(movie_id)

        recommended.append({
            "title": movies_list.iloc[i[0]].title,
            "movie_id": movie_id,
            "poster": meta["poster_url"],
            "details": meta["details"],
            "rating_5": meta["rating_5"],
            "rating_10": meta["rating_10"],
            "year": meta["year"],
        })

    print(f"DEBUG: recommend_content for '{movie}' returned {len(recommended)} items.")
    return recommended


# -------- COLLABORATIVE FILTERING ----------
def recommend_collab(movie_name):
    """
    Recommend similar movies using collaborative filtering.
    """
    if movie_name not in pt.index:
        print(f"WARNING: '{movie_name}' not found in pt index.")
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

        meta = fetch_movie_data(movie_id)

        recommendations.append({
            "title": title,
            "movie_id": movie_id,
            "poster": meta["poster_url"],
            "details": meta["details"],
            "rating_5": meta["rating_5"],
            "rating_10": meta["rating_10"],
            "year": meta["year"],
        })

    print(f"DEBUG: recommend_collab for '{movie_name}' returned {len(recommendations)} items.")
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
    error = None
    model_loaded = similarity_content is not None

    if not model_loaded:
        error = "Content-based recommendation model is not available on the server (similarity.pkl failed to load)."
    elif request.method == "POST":
        selected_movie = request.form.get("movie_name")
        print(f"DEBUG: /content POST, selected_movie = {selected_movie!r}")
        recommendations = recommend_content(selected_movie)
        if not recommendations:
            error = "No recommendations could be generated. Please try another movie."

    return render_template(
        "content_recommend.html",
        movie_list=movie_list_titles,
        selected_movie=selected_movie,
        recommendations=recommendations,
        error=error,
        model_loaded=model_loaded
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
