# 🎬 Smart Movie Recommendation System

A content-based movie recommendation system built using Python and Machine Learning concepts. It recommends movies similar to a user's selection by comparing movie overviews, genres, cast, and directors.

## ✨ Features

* Combines Bollywood and Hollywood movie datasets.
* Includes more than 7,000 movies.
* Uses movie overview, genre, cast, and director information.
* Uses TF-IDF Vectorization to represent movie information numerically.
* Uses Cosine Similarity to identify similar movies.
* Allows users to enter a movie title and receive recommendations.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorization
* Cosine Similarity

## 📂 Project Structure

```text
Smart Movie Recommendation/
├── data/
│   ├── tmdb_5000_movies.csv
│   ├── tmdb_5000_credits.csv
│   └── bollywood_movies.csv
├── movie_recommender.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ How It Works

1. Loads Bollywood and Hollywood movie datasets.
2. Cleans and combines movie information.
3. Extracts useful information about genres, cast, and directors.
4. Combines the movie overview and metadata.
5. Uses TF-IDF to convert text into numerical features.
6. Uses Cosine Similarity to calculate similarity between movies.
7. Displays the most similar movies for the title entered by the user.

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd "Smart Movie Recommendation"
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Run the program

```bash
python movie_recommender.py
```

### 5. Enter a movie name

For example:

```text
Enter a movie name: Jawan
```

The program displays five similar movie recommendations.

## 📊 Example

For the movie **Jawan**, example recommendations include:

* Happy New Year
* Pathaan
* Om Shanti Om
* Chennai Express
* Finding Fanny

Recommendations depend on the dataset and the features used by the model.

## 🎯 Project Objective

The goal of this project is to explore content-based recommendation systems and understand how text processing and similarity algorithms can be used to suggest movies.

## 👩‍💻 Author

Computer Science and Engineering (Data Science) Student

## 📌 Future Improvements

* Develop a web-based user interface.
* Add movie posters and ratings.
* Improve recommendation quality and search functionality.
* Add filtering by language, genre, and release year.
