import joblib
import numpy as np
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

from data_preprocessing import load_ratings, load_movies, build_dataset

DATA_FOLDER = '../data/ml-100k'
MODEL_PATH = '../models/recommender.pkl'


def evaluate(model, train, test, item_features):
    precision = precision_at_k(model, test, train_interactions=train, k=5, item_features=item_features).mean()
    recall = recall_at_k(model, test, train_interactions=train, k=5, item_features=item_features).mean()
    auc = auc_score(model, test, train_interactions=train, item_features=item_features).mean()
    return precision, recall, auc


def main():
    ratings = load_ratings(DATA_FOLDER)
    movies = load_movies(DATA_FOLDER)

    dataset, interactions, item_features = build_dataset(ratings, movies)
    print('Interactions matrix shape:', interactions.shape)

    train, test = random_train_test_split(
        interactions, test_percentage=0.2, random_state=np.random.RandomState(42)
    )

    # Model 1: sirf collaborative (item features nahi)
    cf_model = LightFM(loss='warp', no_components=30, random_state=42)
    cf_model.fit(train, epochs=30, num_threads=2)
    cf_scores = evaluate(cf_model, train, test, None)

    # Model 2: hybrid (genre features ke saath)
    hybrid_model = LightFM(loss='warp', no_components=30, random_state=42)
    hybrid_model.fit(train, item_features=item_features, epochs=30, num_threads=2)
    hybrid_scores = evaluate(hybrid_model, train, test, item_features)

    print('Collaborative -> Precision@5: %.4f | Recall@5: %.4f | AUC: %.4f' % cf_scores)
    print('Hybrid        -> Precision@5: %.4f | Recall@5: %.4f | AUC: %.4f' % hybrid_scores)

    # jiska precision@5 zyada, wahi best
    if hybrid_scores[0] >= cf_scores[0]:
        best_name = 'Hybrid'
        best_model = hybrid_model
        use_features = item_features
    else:
        best_name = 'Collaborative'
        best_model = cf_model
        use_features = None

    print('Best model:', best_name)

    user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()

    joblib.dump(
        {
            'model': best_model,
            'item_features': use_features,
            'user_id_map': user_id_map,
            'item_id_map': item_id_map,
            'model_name': best_name,
        },
        MODEL_PATH,
        compress=3
    )
    print('Model saved at', MODEL_PATH)


if __name__ == '__main__':
    main()