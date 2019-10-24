import pandas as pd
from datetime import datetime

ratings_csv_raw = pd.read_csv('../data/interim/ratings.csv')

'''
1. parse timestamp into date format for ratings
'''
years, months, days = [], [], []
for i, row in ratings_csv_raw.iterrows():
    ts = row.ts
    dt_obj = datetime.fromtimestamp(ts)
    years.append(dt_obj.year)
    months.append(dt_obj.month)
    days.append(dt_obj.day)

ratings_csv_raw['ts_year'] = pd.Series(years)
ratings_csv_raw['ts_month'] = pd.Series(months)
ratings_csv_raw['ts_day'] = pd.Series(days)
ratings_csv_raw.to_csv(index=False, path_or_buf="../data/interim/ratings_with_dates.csv")
print "done."


'''
2. extract year from movie title
'''
mv_year = []
movie_raw = pd.read_csv('../data/interim/movies.csv')
for i, row in movie_raw.iterrows():
    title = row.title
    if '(' in title and ')' in title:
        year = title[title.rfind('(') + 1: title.rfind(')')]
        print year, i
        mv_year.append(int(year))
    else:
        mv_year.append("")
movie_raw['movie_year'] = pd.Series(mv_year)
movie_raw.to_csv(index=False, path_or_buf="../data/interim/movies_with_years.csv")
print "done"
