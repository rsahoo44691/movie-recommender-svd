"""
Gradio screen for the movie recommender.

Author: Rajesh Kumar Sahoo
Course: MSAI-631 - AI for Human-Computer Interaction

This is the web screen for the recommender. I designed it around three simple
ideas from the trust chapter of the course book:
  - Tell the user why a movie was picked.
  - Show an easy label (Strong / Good / Fair match) instead of a raw number.
  - Let the user stay in control: choose how many results and filter by genre.
The screen also tells new users when it cannot personalise yet, so it does not
pretend to know more than it does.

Run:  python3 app.py
"""

from __future__ import annotations

import pandas as pd
import gradio as gr

from src.recommender import build_default

print("Loading the data and fitting the model (the first run takes a few seconds)...")
REC = build_default()
RMSE = REC.evaluate_rmse()
print(f"Model ready. Hold-out RMSE = {RMSE:.3f}")

GENRES = ["Any"] + REC.all_genres
SAMPLE_USERS = REC.known_user_ids[:50]


def _to_frame(recs) -> pd.DataFrame:
    # Turn the list of recommendations into a table the screen can show.
    if not recs:
        return pd.DataFrame(
            columns=["Title", "Match", "Predicted", "Genres", "Why recommended"]
        )
    return pd.DataFrame(
        [
            {
                "Title": r.title,
                "Match": r.match_label,
                "Predicted": r.predicted_rating,
                "Genres": r.genres,
                "Why recommended": r.explanation,
            }
            for r in recs
        ]
    )


# ----------------------------------------------------------------- callbacks
def recommend_for_user(user_id: int, n: int, genre: str):
    # Runs when the user clicks Recommend on the first tab.
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return _to_frame([]), "Please choose a valid user id."
    recs, note = REC.recommend_for_user(user_id, n=int(n), genre=genre)
    status = note or (
        f"Personalised for user {user_id}. These are guessed ratings, not a "
        "promise, so feel free to skip anything that does not fit."
    )
    return _to_frame(recs), status


def search_titles(query: str):
    # Runs when the user searches for a movie title on the second tab.
    matches = REC.search_movies(query, limit=30)
    choices = [f"{title}  ::{mid}" for mid, title in matches]
    update = gr.update(choices=choices, value=choices[0] if choices else None)
    msg = (
        f"{len(choices)} title(s) found. Pick one, then press Find similar."
        if choices
        else "No titles matched. Try another word."
    )
    return update, msg


def similar_for_movie(choice: str, n: int):
    # Runs when the user asks for movies similar to the one they picked.
    if not choice or "::" not in choice:
        return _to_frame([]), "Search and select a movie first."
    movie_id = int(choice.split("::")[-1])
    recs = REC.similar_movies(movie_id, n=int(n))
    title = choice.split("  ::")[0]
    status = (
        f"Movies that are close to '{title}' in taste. The number here is a "
        "similarity score from 0 to 1, not a star rating."
    )
    return _to_frame(recs), status


# ----------------------------------------------------------------------- UI
INTRO = f"""
# Movie Recommender (SVD)

This app suggests movies using SVD on the MovieLens data (610 users and 9,724
movies). It started from the "Movie Recommender with SVD and Flask" tutorial and
was rebuilt here with a Gradio screen and a few features that help the user
trust the results.

**What it can do:** suggest movies for an existing user, or find movies similar
to one you like. **What it cannot do:** it does not know movies outside this
dataset, and it cannot personalise for a brand new user until they rate
something. *Accuracy: hold-out RMSE = {RMSE:.3f} on a 0.5 to 5 rating scale.*
"""

with gr.Blocks(title="Movie Recommender (SVD)", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)

    with gr.Tab("Recommend for a user"):
        gr.Markdown(
            "Pick a user id and the app guesses which unseen movies they would "
            "rate highest. Each row shows why it was picked and a simple match "
            "label instead of a raw score."
        )
        with gr.Row():
            user_dd = gr.Dropdown(
                choices=[str(u) for u in SAMPLE_USERS],
                value=str(SAMPLE_USERS[0]),
                label="User id",
                info="A sample of users the model already knows.",
            )
            n_slider = gr.Slider(1, 20, value=10, step=1, label="How many results")
            genre_dd = gr.Dropdown(
                choices=GENRES, value="Any", label="Filter by genre"
            )
        user_btn = gr.Button("Recommend", variant="primary")
        user_status = gr.Markdown()
        user_table = gr.Dataframe(wrap=True, label="Recommendations")
        user_btn.click(
            recommend_for_user,
            inputs=[user_dd, n_slider, genre_dd],
            outputs=[user_table, user_status],
        )

    with gr.Tab("If you liked..."):
        gr.Markdown(
            "No user id needed. Search a movie you enjoyed and the app returns "
            "the closest movies in taste. This makes it easy to see why a movie "
            "was suggested, through shared genres and a shared audience."
        )
        with gr.Row():
            search_box = gr.Textbox(
                label="Search a movie title", placeholder="e.g. Toy Story"
            )
            search_btn = gr.Button("Search")
        movie_dd = gr.Dropdown(choices=[], label="Select a movie")
        n_slider2 = gr.Slider(1, 20, value=10, step=1, label="How many results")
        sim_btn = gr.Button("Find similar", variant="primary")
        sim_status = gr.Markdown()
        sim_table = gr.Dataframe(wrap=True, label="Similar movies")

        search_btn.click(search_titles, inputs=search_box, outputs=[movie_dd, sim_status])
        search_box.submit(search_titles, inputs=search_box, outputs=[movie_dd, sim_status])
        sim_btn.click(
            similar_for_movie, inputs=[movie_dd, n_slider2], outputs=[sim_table, sim_status]
        )

    gr.Markdown(
        "---\n*These are guesses, not certainties. You stay in control: filter, "
        "skip, or ignore any suggestion.*"
    )


if __name__ == "__main__":
    demo.launch()
