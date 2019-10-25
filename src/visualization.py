import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

ratings_csv = pd.read_csv('../data/interim/ratings_with_dates.csv')
print ratings_csv.head()

print "Unique user_id count: %s" % str(ratings_csv.user_id.nunique())
print "Unique movie_id count: %s" % str(ratings_csv.movie_id.nunique())


# frequency distribution for all the ratings
sns.set(rc={'figure.figsize': (10, 7)})
sns.set_style('whitegrid')
ax = sns.countplot(x='rating', data=ratings_csv, palette=sns.color_palette('Greys'))
ax.set(xlabel='rating', ylabel='count')
ax.set_title("Frequency distribution for all ratings")
plt.show()

# distribution of count of ratings per movie
movie_rating_count = ratings_csv.groupby('movie_id')['rating'].count()
ax = sns.kdeplot(movie_rating_count, shade=True, color='grey')
ax.set_title("Distribution of number of ratings per movie")
plt.show()

# distribution of ratings std
unreliability_ratings = ratings_csv.groupby('movie_id')['rating'].std()
unreliability_ratings.dropna(inplace=True)
ax = sns.kdeplot(unreliability_ratings, shade=True, color='grey')
ax.set_title("Distribution of rating std")
plt.show()


# Pearson correlation
movies = pd.read_csv('../data/processed/movies.csv')
corr_columns = ["genres", "movie_year", "#_rating", "avg_rating", "rating_std", "#_tag", "#_tag_distinct", "avg_tag_frq"]
corr = movies[corr_columns].apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1).round(decimals=2)
print "corr: {}".format(corr)
colormap = plt.cm.RdBu
plt.figure(figsize=(14, 7))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(corr, linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.xticks(rotation=45, fontsize=7)

plt.show()


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


movies = pd.read_csv('../data/processed/movies.csv')
movies = movies[movies['#_rating'] > 10]
columns_to_drop = ["movie_id", "title", "genres"]
movies.drop(columns=columns_to_drop, inplace=True, axis=1)

plot_silhouette_score(movies)
plot_silhouette_histogram(movies)

