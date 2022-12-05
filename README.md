# Movie-recommendation-and-clustering

This repo solves two questions:
1.Investigate recommender performance for different user groups.
Implementing alternating least squares (ALS) to recommend movies for hot users (users frequently rating movies) and cold users (users rarely rating movies) and examine the evaluation results.

2.Find the most popular movie tag.
To find out what the popular tags(or opinions) towards the movies are among all the users, K-means clustering is used on the latent factors developed from ALS to cluster movies, then the most popular tag can be found in the largest cluster.

ratings and tags data are from [MovieLens](https://grouplens.org/datasets/movielens/25m/).

