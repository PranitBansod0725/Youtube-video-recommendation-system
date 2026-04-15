from flask import Flask, render_template, request
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

videos = pd.read_csv("videos.csv")

videos["content"] = videos["title"] + " " + videos["description"]

tfidf = TfidfVectorizer(stop_words="english")
matrix = tfidf.fit_transform(videos["content"])

similarity = cosine_similarity(matrix)

# ==================================
# REPLACED RECOMMEND FUNCTION
# ==================================
def recommend(search):
    search = search.lower().strip()

    # Find matching rows directly
    matched = videos[
        videos["content"].str.lower().str.contains(search, na=False)
    ]

    if matched.empty:
        return ["No Result Found"]

    # Return first 5 matched titles
    return matched["title"].head(5).tolist()

@app.route("/", methods=["GET","POST"])
def home():
    output = []

    if request.method == "POST":
        user = request.form["video"]
        output = recommend(user)

    return render_template("index.html", data=output)

if __name__ == "__main__":
    app.run(debug=True)