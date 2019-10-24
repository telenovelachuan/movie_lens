import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ratings = pd.read_csv('../data/interim/ratings_with_dates.csv')
movies = pd.read_csv('../data/interim/movies_with_years.csv')
tags = pd.read_csv('../data/interim/tags.csv')

'''
1. add average rating, rating count for each movie
'''
# movie_avg_rating = ratings.groupby('movie_id')['rating'].mean().to_frame()
# movie_rating_count = ratings.groupby('movie_id')['rating'].count().to_frame()
#
# movies_with_ratings = movies.merge(movie_avg_rating, on="movie_id", how="left")
# movies_with_ratings = movies_with_ratings.merge(movie_rating_count, on="movie_id", how="left")
# movies_with_ratings.to_csv(index=False, path_or_buf="../data/interim/movies_with_ratings.csv")
# print "done."




'''
2. add tag frequency for every tag
'''
# tag_frq_dict = tags["tag"].value_counts().to_dict()
# tag_frqs = []
# for i, row in tags.iterrows():
#     tag_frq = 0 if row.tag not in tag_frq_dict else tag_frq_dict[row.tag]
#     tag_frqs.append(tag_frq)
#
# tags['tag_frq'] = pd.Series(tag_frqs)
# tags.to_csv(index=False, path_or_buf="../data/interim/tags_with_frq.csv")
#
# print "done"

'''
3. add number of tags for each movie
'''
# movies_with_ratings = pd.read_csv('../data/interim/movies_with_ratings.csv')
# tags = pd.read_csv('../data/interim/tags_with_frq.csv')
# movie_tags = tags.groupby('movie_id')['tag'].count().to_frame()
# movie_tags_distinct = tags.groupby(('movie_id'))['tag'].nunique().to_frame()
# movie_tags_avg_frq = tags.groupby(('movie_id'))['tag_frq'].mean().to_frame()
# movies_with_tags = movies_with_ratings.join(movie_tags, on="movie_id")
# movies_with_tags = movies_with_tags.merge(movie_tags_distinct, on="movie_id", how="left")
# movies_with_tags = movies_with_tags.merge(movie_tags_avg_frq, on="movie_id", how="left")
# movies_with_tags.fillna("0", inplace=True)
# print movies_with_tags
# movies_with_tags.to_csv(index=False, path_or_buf="../data/interim/movies_with_tags.csv")
# print "done."

'''
4. add reliability of ratings and tags
'''
# movie = pd.read_csv('../data/interim/movies_with_tags.csv')
# ratings_std = ratings.groupby('movie_id')['rating'].std()
# movies_with_rating_urb = movie.merge(ratings_std, on='movie_id')
# movies_with_rating_urb.fillna(0, inplace=True)
# movies_with_rating_urb.to_csv(index=False, path_or_buf="../data/interim/movies_with_rating_urb.csv")
# print "done."

'''
5. split and categorize genres
'''
movie = pd.read_csv('../data/interim/movies_with_rating_urb.csv')
genres = set()
for i, row in movie.iterrows():
    genres.update(row.genres.split("|"))

genre_feature_dict = {}
for genre in genres:
    genre_feature_dict[genre] = []
print "genres:{}".format(genres)
for i, row in movie.iterrows():
    for genre in genres:
        if genre in row.genres:
            genre_feature_dict[genre].append(1)
        else:
            genre_feature_dict[genre].append(0)
for genre in genre_feature_dict:
    movie['genre_{}'.format("_".join(genre.split(" ")))] = pd.Series(genre_feature_dict[genre])
print movie
movie.to_csv(index=False, path_or_buf="../data/interim/movies_with_genres_encoded.csv")
print "done."








