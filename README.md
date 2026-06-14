# Movie Recommender (SVD)

**Author:** Rajesh Kumar Sahoo
**Course:** MSAI-631 - AI for Human-Computer Interaction

This project is a movie recommender. It uses SVD (Singular Value Decomposition)
on the MovieLens ratings to guess how a person would rate a movie they have not
seen, and then suggests the movies with the highest guessed ratings. The screen
is built with Gradio.

I started from the public tutorial "Building a Movie Recommender Web App from
Scratch with SVD and Flask" (Towards Data Science) and changed several things.
See the "What I changed" section below.

---

## What it does

- Suggests movies for a user the model already knows.
- Finds movies similar to one you already like (no user id needed).
- Shows a short reason for every suggestion, so you know why it appeared.
- Shows an easy label (Strong / Good / Fair match) instead of a raw number.
- Lets you choose how many results to see and filter them by genre.
- Tells new users when it cannot personalise yet, instead of guessing blindly.

## Folder layout

```
movie-recommender-svd/
├── app.py                  # the Gradio screen (run this)
├── src/
│   └── recommender.py      # the SVD engine: fit, recommend, similar, evaluate
├── data/
│   └── ml-latest-small/    # MovieLens data (ratings, movies, and so on)
├── report/                 # the project report
├── requirements.txt
└── README.md
```

## How to run

```bash
# 1. (optional but recommended) make a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. install the packages
pip install -r requirements.txt

# 3. start the app
python3 app.py
# open the local web address that Gradio prints (usually http://127.0.0.1:7860)
```

To run just the engine and see the accuracy and a sample result:

```bash
python3 src/recommender.py
```

## How it works (in short)

1. Build a table of users (rows) by movies (columns), where each cell is a
   rating. Most cells are empty because people rate very few movies.
2. Subtract each person's own average rating so a generous rater and a strict
   rater can be compared fairly.
3. Run SVD and keep the 50 strongest hidden factors. This removes noise and
   lets the model fill in the empty cells.
4. Rebuild the table to get a guessed rating for every movie a user has not
   seen, and recommend the highest ones.

The same movie factors are reused to find similar movies in the "If you liked"
tab. On a 10% hold-out, the model reaches an RMSE of about 0.93 on the 0.5 to 5
star scale.

## What I changed from the tutorial

- Used Gradio for the screen instead of Flask.
- Added the "If you liked this movie" mode that finds similar movies.
- Added a short reason and an easy match label for each result.
- Added a popularity fallback for new users, weighted so a single 5-star
  rating cannot dominate.
- Added an RMSE check so I can report how accurate the model is.

## Data and credit

- **Dataset:** MovieLens latest-small, from GroupLens. Harper, F. M., & Konstan,
  J. A. (2015). The MovieLens datasets: History and context. *ACM Transactions
  on Interactive Intelligent Systems, 5*(4).
  https://grouplens.org/datasets/movielens/
- **Starting example:** "Building a Movie Recommender Web App from Scratch with
  SVD and Flask" (Towards Data Science).
- **Design ideas:** Kore, A. (2022). *Designing Human-Centric AI Experiences*,
  Chapter 4, "Building Trust." Apress.

## License

For coursework and educational use. The MovieLens data follows GroupLens' usage
terms.
