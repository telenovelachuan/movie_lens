A data science project on the MovieLens dataset to explore the combination of features and build clustering models to determin similar movie recommendations using movie ratings, tags and other combined features.

# Feature explorative ideas
- add average rating and rating count for each movie
- add tag frequency for every tag
- add number of tags for each movie
- add reliability of ratings and tags
- split and categorize genres(something like "multiple-hot" encoding)

# Data preprocessing
- for ratings, parse timestamp into date format
- for movies, extract year from movie title
- standardize all numeric data
- shuffling data and split into training dataset and testing dataset

# Clustering process to determine similar movies based on movie ratings, tags and other combined features
models used:

1. DBSCAN
Traind a model based on DBSCAN lib implemented by sklearn. Tuned hyperparameter eps to 10, and min_samples to 5.

2. Mean Shift
utilized MeanShift class by sklearn.clusters. Set cluster_all to false since there could be some "outlier" movies not similar to any other ones.

3. KMeans
Tried KMeans here in order to test the ideal number of clusters. Silhouette score is used here to evaluate the quality of clustering for different cluster numbers.
The Silhouette score chart in reports/figures shows that a cluster number of 13~15 might be the optimal choice.
The silhouette histogram shows that clusters are not clustered so evenly, but, all clusters seem to be good and clear clusters since their silhouette coefficients go beyond the silhouette score for this number of clusters(vertical dashed line).

4. Ward Hierarchical Clustering
Trained an agglomerative clustering model using ward linkage and Euclidean distance.

5. KNeighbors
Used sklearn's KNeighborsClassifier with the number of neighbors set to 5, though it's a classifier. The model outputs the nearest 5 movies for recommendation.

