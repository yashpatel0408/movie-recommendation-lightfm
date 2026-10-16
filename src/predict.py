import joblib
import numpy as np

from data_preprocessing import load_ratings, load_movies, RATING_THRESHOLD

DATA_FOLDER = '../data/ml-100k'
MODEL_PATH = '../models/recommender.pkl'


def recommend(user_id, top_n=5):
    saved = joblib.load(MODEL_PATH)
    model = saved['model']
    item_features = saved['item_features']
    user_id_map = saved['user_id_map']
    item_id_map = saved['item_id_map']

    ratings = load_ratings(DATA_FOLDER)
    movies = load_movies(DATA_FOLDER)

    if user_id not in user_id_map:
        print('User not found:', user_id)
        return

    internal_user = user_id_map[user_id]
    number_of_items = len(item_id_map)

    # model har movie ka score deta hai is user ke liye
    scores = model.predict(internal_user, np.arange(number_of_items), item_features=item_features)

    # jo movies user pehle hi pasand kar chuka hai unko list se hatao
    liked = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] >= RATING_THRESHOLD)]
    already_liked = set(liked['item_id'])

    # internal id se asli movie id wapas nikalne ke liye reverse map
    reverse_item_map = {}
    for real_id, internal_id in item_id_map.items():
        reverse_item_map[internal_id] = real_id

    titles = dict(zip(movies['item_id'], movies['title']))

    ranked_positions = np.argsort(-scores)

    print('Movies user %d already liked (sample):' % user_id)
    for item_id in list(already_liked)[:5]:
        print('  -', titles[item_id])

    print('Top %d recommendations:' % top_n)
    count = 0
    for position in ranked_positions:
        real_id = reverse_item_map[position]
        if real_id in already_liked:
            continue
        print('  -', titles[real_id])
        count = count + 1
        if count == top_n:
            break


if __name__ == '__main__':
    recommend(user_id=1, top_n=5)