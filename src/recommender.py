"""
Movie recommender engine.

Author: Rajesh Kumar Sahoo
Course: MSAI-631 - AI for Human-Computer Interaction

This file builds a movie recommender using SVD (Singular Value Decomposition)
on the MovieLens ratings. The idea is simple: take the table of who rated what,
find the hidden patterns in it, and use those patterns to guess the ratings a
person has not given yet. The movies with the highest guessed ratings become the
recommendations.

I started from the public tutorial "Building a Movie Recommender Web App from
Scratch with SVD and Flask" (Towards Data Science) and changed a few things:
  - I built the screen with Gradio instead of Flask (see app.py).
  - I added a second mode that finds movies similar to one you already like.
  - I added a short reason and an easy-to-read match label for each result,
    so the user understands why a movie showed up.
  - I added an RMSE check so I can report how accurate the model is.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ml-latest-small")


@dataclass
class Recommendation:
    """One recommended movie and the extra details we show next to it."""

    movie_id: int
    title: str
    genres: str
    predicted_rating: float
    match_label: str      # easy label like Strong / Good / Fair match
    explanation: str      # short reason the movie was picked


class MovieRecommender:
    """A movie recommender that uses SVD on the user-movie rating table."""

    def __init__(self, data_dir: str = DATA_DIR, n_factors: int = 50):
        self.data_dir = data_dir
        self.n_factors = n_factors
        self._is_fit = False

    # ----------------------------------------------------------------- data
    def load(self) -> "MovieRecommender":
        # Read the ratings and the movie titles from the dataset.
        ratings_path = os.path.join(self.data_dir, "ratings.csv")
        movies_path = os.path.join(self.data_dir, "movies.csv")
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)

        # Work out how popular each movie is. We use this for new users who
        # have no history yet, and as a simple baseline.
        pop = (
            self.ratings.groupby("movieId")
            .agg(mean_rating=("rating", "mean"), n_ratings=("rating", "count"))
            .reset_index()
        )
        # Pull the average toward the overall mean so a movie with a single
        # 5-star rating does not jump to the top.
        C = pop["n_ratings"].mean()
        m = pop["mean_rating"].mean()
        pop["weighted_rating"] = (
            (pop["n_ratings"] * pop["mean_rating"] + C * m) / (pop["n_ratings"] + C)
        )
        self.popularity = pop.merge(self.movies, on="movieId")
        return self

    # ------------------------------------------------------------------ fit
    def fit(self) -> "MovieRecommender":
        """Run SVD on the rating table to learn the hidden patterns."""
        # Rows are users, columns are movies, and each cell is a rating.
        # Most cells are empty because people rate very few movies.
        matrix = self.ratings.pivot(
            index="userId", columns="movieId", values="rating"
        )
        self.user_ids = matrix.index.to_numpy()
        self.movie_ids = matrix.columns.to_numpy()
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_index = {m: i for i, m in enumerate(self.movie_ids)}

        R = matrix.to_numpy(dtype=np.float64)
        self.rated_mask = ~np.isnan(R)

        # Some people rate high and some rate low. To compare them fairly,
        # subtract each person's own average before running SVD.
        self.user_means = np.nanmean(R, axis=1)
        R_filled = np.where(self.rated_mask, R, 0.0)
        R_demeaned = R_filled - self.user_means[:, None] * self.rated_mask

        # SVD splits the table into a small number of hidden factors. Keeping
        # only the strongest 50 removes noise and lets us fill the blanks.
        k = min(self.n_factors, min(R.shape) - 1)
        U, sigma, Vt = svds(R_demeaned, k=k)
        # svds returns the factors weakest-first, so flip them to strongest-first.
        order = np.argsort(sigma)[::-1]
        U, sigma, Vt = U[:, order], sigma[order], Vt[order, :]

        self.U, self.sigma, self.Vt = U, np.diag(sigma), Vt
        # Rebuild the full table to get a guessed rating for every user-movie pair.
        self.predicted = (U @ np.diag(sigma) @ Vt) + self.user_means[:, None]
        # Each movie now has its own set of factor values. We use these to find
        # movies that are similar to one another.
        self.movie_factors = Vt.T  # shape: (n_movies, k)
        self._is_fit = True
        return self

    # --------------------------------------------------------- helper lookups
    def _title(self, movie_id: int) -> str:
        row = self.movies.loc[self.movies["movieId"] == movie_id]
        return row["title"].iloc[0] if len(row) else f"Movie {movie_id}"

    def _genres(self, movie_id: int) -> str:
        row = self.movies.loc[self.movies["movieId"] == movie_id]
        g = row["genres"].iloc[0] if len(row) else ""
        return g.replace("|", ", ") if g and g != "(no genres listed)" else "Unknown"

    @staticmethod
    def _match_label(pred: float) -> str:
        """Turn the guessed rating into a plain label.

        A number like 4.31 is hard to read, so we just say Strong, Good, or
        Fair match. It is easier for the user to understand.
        """
        if pred >= 4.3:
            return "Strong match"
        if pred >= 3.7:
            return "Good match"
        return "Fair match"

    @property
    def all_genres(self) -> list[str]:
        seen = set()
        for g in self.movies["genres"]:
            for part in g.split("|"):
                if part and part != "(no genres listed)":
                    seen.add(part)
        return sorted(seen)

    @property
    def known_user_ids(self) -> list[int]:
        return [int(u) for u in self.user_ids]

    # ------------------------------------------------ user recommendations
    def recommend_for_user(
        self, user_id: int, n: int = 10, genre: str | None = None
    ) -> tuple[list[Recommendation], str]:
        """Give the top movies for one user.

        Also returns a short note that tells the user what to expect, for
        example when the user is new and we cannot personalise yet.
        """
        if not self._is_fit:
            raise RuntimeError("Call .load().fit() before recommending.")

        note = ""
        if user_id not in self.user_index:
            # New user: be honest about it and show popular movies instead.
            note = (
                f"User {user_id} is new, so there is no history to learn from "
                "yet. Showing popular titles for now. The list will become "
                "personal once this user rates a few movies."
            )
            return self._popular(n, genre), note

        uidx = self.user_index[user_id]
        n_rated = int(self.rated_mask[uidx].sum())
        preds = self.predicted[uidx].copy()
        # Do not recommend movies the user has already rated.
        preds[self.rated_mask[uidx]] = -np.inf

        if n_rated < 5:
            note = (
                f"User {user_id} has rated only {n_rated} movie(s), so these "
                "guesses are still rough and will improve with more ratings."
            )

        order = np.argsort(preds)[::-1]
        recs: list[Recommendation] = []
        for j in order:
            if preds[j] == -np.inf:
                continue
            movie_id = int(self.movie_ids[j])
            if genre and genre != "Any" and genre not in self._raw_genres(movie_id):
                continue
            pred = float(np.clip(preds[j], 0.5, 5.0))
            recs.append(
                Recommendation(
                    movie_id=movie_id,
                    title=self._title(movie_id),
                    genres=self._genres(movie_id),
                    predicted_rating=round(pred, 2),
                    match_label=self._match_label(pred),
                    explanation=(
                        "People with a similar taste rated this highly "
                        f"(genres: {self._genres(movie_id)})."
                    ),
                )
            )
            if len(recs) >= n:
                break
        return recs, note

    def _raw_genres(self, movie_id: int) -> str:
        row = self.movies.loc[self.movies["movieId"] == movie_id]
        return row["genres"].iloc[0] if len(row) else ""

    def _popular(self, n: int, genre: str | None = None) -> list[Recommendation]:
        # Fallback list for new users: the best-rated popular movies.
        pop = self.popularity.sort_values("weighted_rating", ascending=False)
        recs: list[Recommendation] = []
        for _, row in pop.iterrows():
            if genre and genre != "Any" and genre not in row["genres"]:
                continue
            recs.append(
                Recommendation(
                    movie_id=int(row["movieId"]),
                    title=row["title"],
                    genres=self._genres(int(row["movieId"])),
                    predicted_rating=round(float(row["weighted_rating"]), 2),
                    match_label="Popular pick",
                    explanation=(
                        f"Rated highly by {int(row['n_ratings'])} viewers "
                        "(shown because we cannot personalise yet)."
                    ),
                )
            )
            if len(recs) >= n:
                break
        return recs

    # ------------------------------------------------ item-item ("if you liked")
    def similar_movies(self, movie_id: int, n: int = 10) -> list[Recommendation]:
        """Find movies that are closest to a chosen movie."""
        if movie_id not in self.movie_index:
            return []
        idx = self.movie_index[movie_id]
        # Compare the chosen movie's factors with every other movie's factors.
        sims = cosine_similarity(
            self.movie_factors[idx][None, :], self.movie_factors
        )[0]
        order = np.argsort(sims)[::-1]
        seed_genres = set(self._raw_genres(movie_id).split("|"))
        recs: list[Recommendation] = []
        for j in order:
            if j == idx:
                continue
            mid = int(self.movie_ids[j])
            shared = seed_genres & set(self._raw_genres(mid).split("|"))
            shared.discard("(no genres listed)")
            reason = (
                f"Shares {', '.join(sorted(shared))} and the same kind of "
                f"audience as '{self._title(movie_id)}'."
                if shared
                else f"Watched by the same kind of audience as "
                     f"'{self._title(movie_id)}'."
            )
            recs.append(
                Recommendation(
                    movie_id=mid,
                    title=self._title(mid),
                    genres=self._genres(mid),
                    predicted_rating=round(float(sims[j]), 2),
                    match_label=self._match_label(2.5 + 2.5 * float(sims[j])),
                    explanation=reason,
                )
            )
            if len(recs) >= n:
                break
        return recs

    def search_movies(self, query: str, limit: int = 20) -> list[tuple[int, str]]:
        """Search movie titles so the user can pick one in the 'if you liked' tab."""
        q = (query or "").strip().lower()
        df = self.movies
        if q:
            df = df[df["title"].str.lower().str.contains(q, na=False)]
        return [(int(r.movieId), r.title) for r in df.head(limit).itertuples()]

    # -------------------------------------------------------------- evaluation
    def evaluate_rmse(self, test_frac: float = 0.1, seed: int = 42) -> float:
        """Check accuracy: hide 10% of ratings, train on the rest, then score.

        RMSE (root mean square error) tells us, on average, how far off the
        guessed ratings are from the real ones.
        """
        ratings = self.ratings.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        cut = int(len(ratings) * (1 - test_frac))
        train, test = ratings.iloc[:cut], ratings.iloc[cut:]

        matrix = train.pivot(index="userId", columns="movieId", values="rating")
        u_ids = matrix.index.to_numpy()
        m_ids = matrix.columns.to_numpy()
        u_idx = {u: i for i, u in enumerate(u_ids)}
        m_idx = {m: i for i, m in enumerate(m_ids)}

        R = matrix.to_numpy(dtype=np.float64)
        mask = ~np.isnan(R)
        means = np.nanmean(R, axis=1)
        R_dem = np.where(mask, R, 0.0) - means[:, None] * mask
        k = min(self.n_factors, min(R.shape) - 1)
        U, sigma, Vt = svds(R_dem, k=k)
        pred = (U @ np.diag(sigma) @ Vt) + means[:, None]

        errors = []
        for r in test.itertuples():
            if r.userId in u_idx and r.movieId in m_idx:
                p = pred[u_idx[r.userId], m_idx[r.movieId]]
                errors.append((np.clip(p, 0.5, 5.0) - r.rating) ** 2)
        return float(np.sqrt(np.mean(errors))) if errors else float("nan")


def build_default() -> MovieRecommender:
    """Load the data and fit the model in one call."""
    return MovieRecommender().load().fit()


if __name__ == "__main__":
    rec = build_default()
    print(f"Users: {len(rec.user_ids)}  Movies: {len(rec.movie_ids)}")
    print(f"Hold-out RMSE: {rec.evaluate_rmse():.4f}")
    recs, note = rec.recommend_for_user(1, n=5)
    if note:
        print("NOTE:", note)
    for r in recs:
        print(f"  {r.predicted_rating:>4}  [{r.match_label}]  {r.title}")
