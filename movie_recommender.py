import pandas as pd
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. FUNCTION TO EXTRACT NAMES
# ==========================================

def extract_names(text, key="name", limit=5):

    try:
        data = ast.literal_eval(text)

        names = []

        for item in data[:limit]:
            if key in item:
                names.append(
                    item[key].replace(" ", "").lower()
                )

        return " ".join(names)

    except:
        return ""


# ==========================================
# 2. LOAD HOLLYWOOD MOVIES
# ==========================================

hollywood = pd.read_csv(
    "data/tmdb_5000_movies.csv"
)

hollywood = hollywood[
    ["title", "overview", "genres"]
]

hollywood = hollywood.rename(
    columns={
        "title": "movie_name"
    }
)


# ==========================================
# 3. LOAD HOLLYWOOD CREDITS
# ==========================================

credits = pd.read_csv(
    "data/tmdb_5000_credits.csv"
)

credits = credits[
    ["title", "cast", "crew"]
]

credits = credits.rename(
    columns={
        "title": "movie_name"
    }
)


# ==========================================
# 4. COMBINE HOLLYWOOD DATA
# ==========================================

hollywood = hollywood.merge(
    credits,
    on="movie_name",
    how="left"
)


# ==========================================
# 5. CLEAN HOLLYWOOD GENRES
# ==========================================

hollywood["genre"] = hollywood["genres"].apply(
    lambda x: extract_names(x)
)


# ==========================================
# 6. CLEAN HOLLYWOOD CAST
# ==========================================

hollywood["cast"] = hollywood["cast"].apply(
    lambda x: extract_names(x, "name", 5)
)


# ==========================================
# 7. EXTRACT HOLLYWOOD DIRECTOR
# ==========================================

def get_director(text):

    try:

        data = ast.literal_eval(text)

        for person in data:

            if (
                person.get("job") == "Director"
            ):

                return person.get(
                    "name",
                    ""
                ).replace(" ", "").lower()

    except:
        return ""

    return ""


hollywood["director"] = hollywood["crew"].apply(
    get_director
)


# ==========================================
# 8. KEEP REQUIRED COLUMNS
# ==========================================

hollywood = hollywood[
    [
        "movie_name",
        "overview",
        "genre",
        "director",
        "cast"
    ]
]


# ==========================================
# 9. LOAD BOLLYWOOD DATA
# ==========================================

bollywood = pd.read_csv(
    "data/bollywood_movies.csv"
)

bollywood = bollywood[
    [
        "movie_name",
        "overview",
        "genre",
        "director",
        "cast"
    ]
]


# ==========================================
# 10. CLEAN BOLLYWOOD DATA
# ==========================================

columns = [
    "movie_name",
    "overview",
    "genre",
    "director",
    "cast"
]

for column in columns:

    bollywood[column] = (
        bollywood[column]
        .fillna("")
        .astype(str)
    )


# ==========================================
# 11. COMBINE BOTH DATASETS
# ==========================================

movies = pd.concat(
    [
        hollywood,
        bollywood
    ],
    ignore_index=True
)


# ==========================================
# 12. CLEAN ALL DATA
# ==========================================

for column in columns:

    movies[column] = (
        movies[column]
        .fillna("")
        .astype(str)
    )


print(
    "Hollywood movies:",
    len(hollywood)
)

print(
    "Bollywood movies:",
    len(bollywood)
)

print(
    "Total movies:",
    len(movies)
)


# ==========================================
# 13. CREATE COMBINED FEATURES
# ==========================================

movies["combined_features"] = (

    movies["overview"] + " " +

    movies["genre"] + " " +

    movies["director"] + " " +

    movies["cast"]
)


# ==========================================
# 14. CREATE TF-IDF
# ==========================================

tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    movies["combined_features"]
)


print(
    "\nTF-IDF matrix created!"
)

print(
    "Matrix shape:",
    tfidf_matrix.shape
)


# ==========================================
# 15. CALCULATE SIMILARITY
# ==========================================

similarity = cosine_similarity(
    tfidf_matrix
)


print(
    "\nSimilarity matrix created!"
)

print(
    "Similarity matrix shape:",
    similarity.shape
)


# ==========================================
# 16. RECOMMENDATION FUNCTION
# ==========================================

def recommend(
    movie_name,
    number_of_recommendations=5
):

    # Find movie

    movie_matches = movies[
        movies["movie_name"]
        .str.strip()
        .str.lower()
        ==
        movie_name.strip().lower()
    ]


    # Movie not found

    if movie_matches.empty:

        print(
            "\nMovie not found!"
        )

        return


    # Get movie index

    movie_index = (
        movie_matches.index[0]
    )


    # Get similarity scores

    similarity_scores = list(
        enumerate(
            similarity[
                movie_index
            ]
        )
    )


    # Sort highest similarity first

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )


    print(
        f"\nMovies similar to "
        f"'{movie_matches.iloc[0]['movie_name']}':"
    )


    count = 0


    for index, score in similarity_scores:

        # Don't recommend same movie

        if index == movie_index:
            continue


        print(
            f"{count + 1}. "
            f"{movies.iloc[index]['movie_name']} "
            f"(similarity: {score:.2f})"
        )


        count += 1


        if (
            count
            ==
            number_of_recommendations
        ):

            break


# ==========================================
# 17. TEST
# ==========================================

# ==========================================
# 17. USER INPUT
# ==========================================

movie_name = input(
    "\nEnter a movie name: "
)

recommend(
    movie_name,
    5
)