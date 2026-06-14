# Building a Movie Recommendation System with SVD and a Human-Centered Interface

*AI for Human–Computer Interaction (MSAI-631)*

---

> **Formatting note for the Word version:** This draft supplies the title-page
> content, body, and references. In Microsoft Word, apply APA 7th edition
> formatting: a separate title page, 12-pt Times New Roman, double spacing,
> 1-inch margins, a running head/page number, and a hanging-indent References
> page. The body below runs about five to seven pages once double-spaced
> (title and reference pages excluded).

**Title:** Building a Movie Recommendation System with SVD and a Human-Centered Interface
**Author:** [Your Name]
**Course:** MSAI-631 – Artificial Intelligence for Human–Computer Interaction
**Institution:** University of the Cumberlands
**Date:** [Submission Date]
**GitHub repository:** [paste repository URL here]

---

## Abstract

Recommendation systems shape much of how people encounter information online,
from the films that appear on a streaming service to the products suggested on
a retail page. This report documents the design and construction of a movie
recommendation system that uses Singular Value Decomposition (SVD) to predict
how a user would rate films they have not yet seen. The work began from a
publicly available tutorial that paired SVD with a Flask web application and
extended it in two directions: a different, more interactive front end built
with Gradio, and a set of interface features drawn from human–computer
interaction (HCI) research on trust. The finished system explains why each
movie is recommended, expresses its certainty in plain language rather than a
raw number, and leaves the user in control of how results are filtered. On a
held-out sample of ratings the model reached a root-mean-square error (RMSE) of
roughly 0.93 on a five-point scale. The report defines recommendation systems,
describes their theoretical foundations and practical applications, walks
through the construction process, and reflects on the challenges encountered and
on how conversational and proactive assistants may evolve.

## Introduction: What a Recommendation System Is

A recommendation system is software that predicts which items a person is most
likely to value and then presents those items to them. Instead of waiting for a
user to search, the system anticipates interest based on past behavior, the
behavior of similar people, or the properties of the items themselves (Aggarwal,
2016). These systems have become a central part of everyday digital life because
catalogs are now far too large to browse by hand; a viewer cannot scroll through
every title a streaming service offers, so the service ranks a handful that it
believes will appeal to that particular viewer.

The goal of this project was to define recommendation systems clearly, build a
working example, and connect the engineering to the human side of the
experience. The first half of that goal is technical: an algorithm must turn a
sparse record of past ratings into reliable predictions. The second half is a
question of interaction design. A recommendation is only useful if a person
trusts it enough to act on it, understands why it appeared, and can override it
when it is wrong. Kore (2022) argues that building the right level of trust is a
core part of designing any AI product, and that two ingredients matter most:
explainability and user control. Those ideas guided the interface for this
system.

## Theoretical Foundations

Recommendation systems are usually grouped into three broad approaches.
**Content-based filtering** recommends items that resemble those a user already
liked, using item features such as genre or keywords (Ricci et al., 2015).
**Collaborative filtering** ignores item features and instead looks for patterns
across many users: if people who rated the same movies as you also enjoyed a
film you have not seen, that film becomes a good candidate (Koren et al., 2009).
**Association-rule mining**, the third approach, finds items that frequently
appear together in transactions and underlies the familiar "frequently bought
together" suggestion; the Apriori algorithm is its classic implementation
(Agrawal & Srikant, 1994).

This project uses collaborative filtering implemented with matrix
factorization. The starting point is a large table whose rows are users, whose
columns are movies, and whose cells hold ratings. The table is extremely
**sparse**, because any individual has rated only a tiny fraction of the
catalog. Matrix factorization addresses this by assuming that both users and
movies can be described by a small number of hidden, or *latent*, factors. One
latent factor might loosely correspond to how much action a film contains,
another to how light or serious it is; the algorithm discovers these dimensions
on its own rather than being told what they mean (Koren et al., 2009).

Singular Value Decomposition is the mathematical tool that recovers those
factors. SVD expresses the rating matrix *R* as the product of three matrices,
*R ≈ U Σ Vᵀ*, where *U* describes users in terms of latent factors, *V*
describes movies in the same terms, and *Σ* weights how important each factor
is. By keeping only the strongest *k* factors (here, fifty), the method
compresses the data, filters out noise, and—crucially—produces sensible
estimates for the empty cells. Multiplying the reduced matrices back together
yields a predicted rating for every user–movie pair, including pairs the user
has never encountered. The highest predicted ratings for unseen movies become
the recommendations.

## Practical Applications

Collaborative filtering and matrix factorization are not academic curiosities;
they power large parts of the digital economy. The approach rose to prominence
during the Netflix Prize competition, where matrix-factorization models proved
to be among the most accurate predictors of film ratings (Koren et al., 2009).
The same family of techniques drives product suggestions on e-commerce sites,
playlist generation on music services, video recommendations, and job and
connection suggestions on professional networks. Beyond entertainment and
retail, recommendation logic supports decision support in healthcare, education,
and news, where it helps people find relevant material without exhaustive
searching. Because these systems increasingly influence what people see and buy,
researchers have begun to stress the importance of building human values, and
not only engagement, into their design (Stray et al., 2022).

## Method: Building the System

### Data

The project uses the MovieLens *latest-small* dataset published by GroupLens
(Harper & Konstan, 2015). It contains roughly 100,000 ratings from 610 users
across 9,724 movies, along with a separate file of movie titles and genres. The
dataset is widely used for teaching and benchmarking, which made it a good fit
for an educational project and allowed the results to be compared informally
with published work.

### The recommendation engine

The engine was written in Python using NumPy, pandas, SciPy, and scikit-learn.
The ratings file is pivoted into a user-by-movie matrix. Each user's ratings are
then centered on that user's own average, so that a generous rater and a harsh
rater are placed on a common scale before factorization. SciPy's sparse SVD
routine factors the centered matrix into fifty latent factors, and the predicted
rating matrix is reconstructed by recombining the factors and adding each user's
average back in. To generate recommendations for a person, the system looks up
that person's row of predictions, removes movies they have already rated, and
returns the highest-scoring remainder.

The same latent factors were reused for a second feature. Because every movie is
now described by a short vector of factor values, the cosine similarity between
two movie vectors measures how alike they are in taste space. This supports an
"if you liked this movie" mode that needs no user identity at all and is easy to
explain to a user in terms of shared genres and shared audiences.

### The interface

Rather than reproduce the original tutorial's Flask application, the interface
was rebuilt with Gradio, which generates an interactive web UI directly from
Python functions. The interface presents two tabs: one for personalized
recommendations by user, and one for the "if you liked" similarity search.
Several design choices were made deliberately to support trust, following
Kore (2022):

- **Explainability.** Every recommended row carries a short, plain-language
  reason, such as that users with similar taste rated the film highly, or that
  it shares specific genres and audiences with a film the user selected. The
  application also states up front what it can and cannot do, which helps users
  form an accurate mental model.
- **Calibrated confidence.** Instead of showing a raw decimal prediction, which
  is hard for people to interpret, each recommendation is labeled "Strong,"
  "Good," or "Fair" match. Kore (2022) notes that a friendly category often
  helps users calibrate trust better than a precise-looking number.
- **User control.** Users choose how many results to see and can filter by
  genre, and they can ignore or dismiss any suggestion. Control of this kind
  makes people more willing to keep using a system even when it occasionally
  errs.
- **Setting expectations.** When a user has rated very few movies, or is not in
  the dataset at all, the system says so and falls back to popular titles rather
  than presenting weak guesses as if they were confident predictions.

### Evaluation

To gauge accuracy, ten percent of the ratings were held out, the model was
refit on the remaining ninety percent, and predictions for the held-out ratings
were compared against the true values. The resulting RMSE was approximately
0.93 on the half-star-to-five-star scale, which is in the range typically
reported for straightforward SVD models on this dataset and is comfortably
better than guessing the average rating for every film.

## Modifications and What They Demonstrate

The assignment asked for meaningful changes to the base example rather than a
copy. The clearest change is the interface technology: moving from Flask to
Gradio required restructuring the application around callback functions and
state, which demonstrated an understanding of how the recommender's outputs flow
into a UI. The second change is the addition of the item-item similarity mode,
which reuses the learned movie factors in a way the original tutorial did not.
The third and most substantial change is the layer of human-centered trust
features, which connect the technical output to the HCI concepts at the heart of
the course. A smaller but important refinement was the use of a
shrinkage-weighted popularity score for the cold-start fallback, so that a movie
with a single five-star rating cannot outrank a film with hundreds of strong
ratings.

## Challenges and Mitigation Strategies

Several practical obstacles arose during construction. The first was a
dependency conflict: the most recent version of the Gradio library no longer
supports the older Python interpreter that was available, and an automatic
installation pulled in a newer support library that had removed a function the
interface code expected. This was resolved by pinning Gradio to its last
version compatible with the interpreter and pinning the support library to a
matching release. The exact versions are recorded in the project's requirements
file so the environment can be reproduced.

A second challenge was the sparsity of the rating matrix. With most cells empty,
naive averages are unstable and cold-start users have no history to learn from.
Centering each user's ratings before factorization, clipping predictions to the
valid rating range, and providing an explicit popularity-based fallback for new
users together addressed these issues and, helpfully, turned a technical
limitation into an opportunity to set honest expectations in the interface.

A third challenge was interpretability. Matrix factorization produces accurate
numbers but no built-in reason a person can read. The mitigation was to attach
human-readable explanations derived from information the model and dataset
already contain, such as shared genres and the fact that similar users rated a
film highly, so that explanations remain truthful rather than decorative.

## Reflection: Will Assistants Become Proactive?

The project brief asks whether people will always type questions into assistants
or whether those assistants will become more proactive. The recommendation
system built here is already a small step away from pure question-and-answer
interaction: the "recommend for you" tab volunteers suggestions without being
asked a specific question. It seems likely that future systems will continue to
shift in this direction, surfacing relevant items and actions in context rather
than waiting for an explicit query. Kore (2022) cautions, however, that
proactivity raises the stakes for trust. A system that acts on its own must be
even clearer about what it is doing, more careful to stay within the bounds of
what it can actually deliver, and more generous with user control, so that
people can correct or switch off behavior they did not request. The most useful
assistants will probably not be the most aggressively proactive ones, but the
ones that calibrate their initiative to how confident they are and how high the
stakes are.

## Conclusion

This project defined recommendation systems, surveyed their theoretical
foundations and practical uses, and built a working movie recommender using SVD
on the MovieLens dataset. Beyond reproducing a known technique, the work
extended a public example with a new interface, an additional recommendation
mode, and a set of trust-building features grounded in HCI research. The
exercise reinforced a lesson that runs through the course: a recommendation
engine and the experience around it are inseparable. An accurate model that
cannot explain itself or be controlled will not be trusted, and an untrusted
recommendation, however accurate, goes unused.

## References

Aggarwal, C. C. (2016). *Recommender systems: The textbook*. Springer.

Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules.
In *Proceedings of the 20th International Conference on Very Large Data Bases
(VLDB)* (pp. 487–499). Morgan Kaufmann.

Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and
context. *ACM Transactions on Interactive Intelligent Systems, 5*(4), 1–19.
https://doi.org/10.1145/2827872

Kore, A. (2022). *Designing human-centric AI experiences: Applied UX design for
artificial intelligence*. Apress.

Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for
recommender systems. *Computer, 42*(8), 30–37.
https://doi.org/10.1109/MC.2009.263

Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender systems: Introduction
and challenges. In F. Ricci, L. Rokach, & B. Shapira (Eds.), *Recommender
systems handbook* (2nd ed., pp. 1–34). Springer.

Stray, J., Halevy, A., Assar, P., Hadfield-Menell, D., Boutilier, C., Ashar, A.,
Bakalar, C., Beattie, L., Ekstrand, M., & Leibowicz, C. (2022). Building human
values into recommender systems: An interdisciplinary synthesis. *ACM
Transactions on Recommender Systems*. https://doi.org/10.1145/3556531
