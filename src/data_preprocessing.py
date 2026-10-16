import pandas as pd
from lightfm.data import Dataset

GENRE_NAMES = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

RATING_THRESHOLD = 4


def load_ratings(folder):
    # u.data: user_id, item_id, rating, timestamp (tab separated)
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(folder + '/u.data', sep='\t', names=columns)
    return ratings


def load_movies(folder):
    # u.item: movie id | title | release date | video release date | url | 19 genre flags
    columns = ['item_id', 'title', 'release_date', 'video_release_date', 'url'] + GENRE_NAMES
    movies = pd.read_csv(folder + '/u.item', sep='|', names=columns, encoding='latin-1')
    return movies


def get_movie_genres(movies):
    # har movie ke liye uski genres ki list banata hai
    genres_of_movie = {}
    for index, row in movies.iterrows():
        genre_list = []
        for genre in GENRE_NAMES:
            if row[genre] == 1:
                genre_list.append(genre)
        genres_of_movie[row['item_id']] = genre_list
    return genres_of_movie


def build_dataset(ratings, movies):
    # sirf 4 ya 5 rating wale ko "pasand aaya" maante hain
    liked = ratings[ratings['rating'] >= RATING_THRESHOLD]

    dataset = Dataset()
    dataset.fit(
        users=ratings['user_id'].unique(),
        items=movies['item_id'].unique(),
        item_features=GENRE_NAMES
    )

    pairs = []
    for index, row in liked.iterrows():
        pairs.append((row['user_id'], row['item_id']))
    interactions, weights = dataset.build_interactions(pairs)

    genres_of_movie = get_movie_genres(movies)
    feature_list = []
    for item_id, genre_list in genres_of_movie.items():
        feature_list.append((item_id, genre_list))
    item_features = dataset.build_item_features(feature_list)

    return dataset, interactions, item_features