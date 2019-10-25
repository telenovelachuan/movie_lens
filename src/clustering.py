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
Explore Silhouette score using KMeans to determine optimal number of clusters
'''


def plot_silhouette_score(X):
    silhouette_scores = []
    from sklearn.metrics import silhouette_score
    for i in range(20)[2:]:
        km = KMeans(n_clusters=i)
        km.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, km.labels_))
    plt.plot(silhouette_scores)
    plt.xticks(range(0, 19))
    plt.xlabel("# of clusters")
    plt.ylabel("silhouette score")
    plt.show()


def plot_silhouette_histogram(X):
    from sklearn.metrics import silhouette_samples
    from matplotlib.ticker import FixedLocator, FixedFormatter
    import matplotlib as mpl
    import numpy as np

    silhouette_scores = []
    from sklearn.metrics import silhouette_score
    for i in range(20)[2:]:
        km = KMeans(n_clusters=i)
        km.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, km.labels_))
    print "begin to plot silhouette histogram... silhouette_scores:{}".format(silhouette_scores)

    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                    for k in range(1, 30)]
    inertias = [model.inertia_ for model in kmeans_per_k]
    plt.figure(figsize=(10, 5))

    for k in (12, 13, 14, 15):
        plt.subplot(2, 2, k - 11)

        y_pred = kmeans_per_k[k - 1].labels_
        silhouette_coefficients = silhouette_samples(X, y_pred)

        padding = len(X) // 30
        pos = padding
        ticks = []
        for i in range(k):
            coeffs = silhouette_coefficients[y_pred == i]
            coeffs.sort()

            color = mpl.cm.Spectral(i / k)
            plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                              facecolor=color, edgecolor=color, alpha=0.7)
            ticks.append(pos + len(coeffs) // 2)
            pos += len(coeffs) + padding

        plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
        plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
        if k in (12, 14):
            plt.ylabel("Cluster")

        if k in (14, 15):
            plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.xlabel("Silhouette Coefficient")
        else:
            plt.tick_params(labelbottom=False)

        plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
        plt.title("$k={}$".format(k), fontsize=16)

    plt.show()


plot_silhouette_score(movies)
plot_silhouette_histogram(movies)


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

