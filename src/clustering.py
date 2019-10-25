import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump


movies = pd.read_csv('../data/processed/movies.csv')
# remove outliers
movies = movies[movies['#_rating'] > 10]
print movies
movies.to_csv("../data/processed/movies.csv")

# standardization
columns_to_drop = ["movie_id", "title", "genres"]
movies.drop(columns=columns_to_drop, inplace=True, axis=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(movies.values)
scaled_features_df = pd.DataFrame(scaled_features, index=movies.index, columns=movies.columns)
print "scaled_features_df:\n{}".format(scaled_features_df)

# shuffling
scaled_features_df = scaled_features_df.sample(frac=1)
split_point = int(len(scaled_features_df) * 0.8)
print "after shuffle:\n {}".format(split_point)

training_set = scaled_features_df[: split_point]
print "training_set:{}".format(training_set)
test_set = scaled_features_df[split_point:]
print "test_set:{}".format(test_set)
print "test_set.as_matrix:{}".format(test_set.as_matrix())
test_set.to_csv("../data/processed/test_set.csv")

'''
NearestNeighbors
'''
from sklearn.neighbors import NearestNeighbors
print "begin to train NearestNeighbors..."
nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(training_set)
print "training finished"
distances, indices = nbrs.kneighbors([test_set.as_matrix()[0]])
print "distances:{}".format(distances)
recommended_movies = []
print "indices:{}".format(indices)
for r in indices:
    recommended_movies.append(movies.iloc[r].index)
print "recommended movies by NearestNeighbors:{}".format(recommended_movies)


'''
KNeighborsClassifier
'''
from sklearn.neighbors import KNeighborsClassifier
print "begin to train KNeighborsClassifier..."
kn_clf = KNeighborsClassifier(n_neighbors=5)
kn_clf.fit(movies.values, movies.index.values)
print "training finished."
dump(kn_clf, '../models/kneighbors.joblib')
print "saving model finished"
indices = kn_clf.kneighbors(X=[test_set.as_matrix()[0]], n_neighbors=5, return_distance=False)
print "indices:{}".format(indices)
recommended_movies = []
for r in indices:
    recommended_movies.append(movies.iloc[r].index)
print "recommended movies by KNeighborsClassifier:{}".format(recommended_movies)


'''
Clustering algorithms
'''


def count_clustered_labels(labels_, algorithm_name):
    from collections import Counter
    print "clustering result by {}\n{}".format(algorithm_name, Counter(labels_))

# DBSCAN
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=11, min_samples=5)
dbscan.fit(movies.values)
dump(dbscan, '../models/dbscan.joblib')
print "saving model finished"
count_clustered_labels(dbscan.labels_, "DBSCAN")

# Mean Shift
from sklearn.cluster import MeanShift
print "training MeanShift..."
ms = MeanShift(cluster_all=False)
ms.fit(movies.values)
print "training finished"
dump(ms, '../models/mean_shift.joblib')
print "saving model finished"
count_clustered_labels(ms.labels_, "Mean Shift")

# Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
agc = AgglomerativeClustering(affinity='euclidean', linkage='ward')
agc.fit(movies.values)
dump(agc, '../models/ward_hierarchical.joblib')
print "saving model finished"
count_clustered_labels(agc.labels_, "Ward hierarchical clustering")

