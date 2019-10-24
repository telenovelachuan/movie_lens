import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ratings_csv = pd.read_csv('../data/interim/ratings_with_dates.csv')
print ratings_csv.head()

print "Unique user_id count: %s" % str(ratings_csv.user_id.nunique())
print "Unique movie_id count: %s" % str(ratings_csv.movie_id.nunique())


# # frequency for all the ratings
# sns.set(rc={'figure.figsize': (10, 7)})
# sns.set_style('whitegrid')
# ax = sns.countplot(x='rating', data=ratings_csv, palette=sns.color_palette('Greys'))
# ax.set(xlabel='rating', ylabel='count')
# plt.show()
#
# # distribution of count of ratings per movie
# movie_rating_count = ratings_csv.groupby('movie_id')['rating'].count()
# ax = sns.kdeplot(movie_rating_count, shade=True, color='grey')
# plt.show()
#
# # distribution of ratings std
# unreliability_ratings = ratings_csv.groupby('movie_id')['rating'].std()
# unreliability_ratings.dropna(inplace=True)
# ax = sns.kdeplot(unreliability_ratings, shade=True, color='grey')
# plt.show()


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

