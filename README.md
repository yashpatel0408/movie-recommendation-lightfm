# Movie Recommendation System using LightFM

Hybrid recommendation system that suggests movies to users, built with LightFM on the MovieLens 100K dataset.

## Dataset
MovieLens 100K (943 users, 1682 movies, 100,000 ratings) from GroupLens.
Download: https://grouplens.org/datasets/movielens/100k/
A rating of 4 or 5 is treated as "user liked the movie".

## Approach
1. EDA: rating distribution, ratings per user and movie, genres, sparsity
2. Interaction matrix of users and liked movies, 80/20 train-test split
3. Collaborative model: LightFM with WARP loss, no item features
4. Hybrid model: same model with movie genres as item features
5. Evaluation: Precision@5, Recall@5, AUC on the test set
6. Recommendations for a user, excluding movies already liked

## Results
- Best model: <naam daalo>
- Precision@5: <value daalo>
- Recall@5: <value daalo>
- AUC: <value daalo>

## Key Insights
(100000, 4) (1682, 24)
- rating
1     6110
2    11370
3    27145
4    34174
5    21201
-             Collaborative  Hybrid
Precision@5         0.2549  0.2593
Recall@5            0.1516  0.1423
AUC                 0.9355  0.9334


## How to run
LightFM does not install easily on Windows with newer Python versions. Run on Linux or Google Colab:

pip install -r requirements.txt
mkdir -p data
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip -d data
cd src
python train_model.py
python predict.py

INTERN ID:-CITS9194

