import pandas as pd
import ast
import random
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.calibration import LabelEncoder

def load_data():
    # Load user ratings
    df_users = pd.read_csv('data/movies_users.csv')
    df_triple = pd.read_csv('data/movies_ratings_cleaned.csv')

    # Load movie titles with genres
    df_titles_genres = pd.read_csv('data/movies_titles_with_genres.csv')
    df_titles_genres['genres'] = df_titles_genres['genres'].apply(ast.literal_eval)
    df_titles_genres.set_index('show_id', inplace=True)

    # Build interaction user-item matrix
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()

    df_triple['user_index'] = user_enc.fit_transform(df_triple['user_id'])
    df_triple['item_index'] = item_enc.fit_transform(df_triple['show_id'])

    X = csr_matrix(
        (df_triple['rating'], (df_triple['item_index'], df_triple['user_index'])),
        shape=(df_triple['item_index'].nunique(), df_triple['user_index'].nunique())
    )

    # Mappers
    item_mapper = dict(zip(df_triple['show_id'], df_triple['item_index']))
    item_inv_mapper = dict(zip(df_triple['item_index'], df_triple['show_id']))

    return df_users, df_triple, df_titles_genres, X, item_mapper, item_inv_mapper

def get_genre_recommendations_for_user(
    age, gender, platforms, genre_list,
    df_users, df_triple, df_titles_genres,
    X, item_mapper, item_inv_mapper,
    k=100, max_recs=5, age_range=5
):
    import difflib

    # Map simplified genre labels to actual genres in the dataset
    genre_aliases = {
        "Comedy": ["Comedies", "TV Comedies", "Comedies Romantic Movies"],
        "Romance": ["Romantic Movies", "Comedies Romantic Movies", "Dramas Romantic Movies"],
        "Action": ["Action", "Adventure", "TV Action"],
        "Drama": ["Dramas", "TV Dramas"],
        "Documentary": ["Documentaries", "Docuseries"],
        "Fantasy": ["Fantasy"],
        "Horror": ["Horror Movies"],
        "Thriller": ["Thrillers"],
        "Kids": ["Children", "Kids' TV", "Family Movies"]
    }

    # Step 1: Filter users by age and gender
    matching_users = df_users[
        (df_users['gender'] == gender) &
        (df_users['age'].between(age - age_range, age + age_range))
    ]
    print(f"After age/gender filter: {len(matching_users)} users")

    # Step 2: Smart platform filtering
    min_users_after_filter = 3
    skipped_platforms = []

    for platform, value in platforms.items():
        before = len(matching_users)
        temp_filtered = matching_users[matching_users[platform] == value]
        after = len(temp_filtered)
        print(f"Trying platform {platform}={value}: {before} → {after}")

        if after >= min_users_after_filter:
            matching_users = temp_filtered
            print(f"Keeping filter {platform}={value}")
        else:
            skipped_platforms.append(platform)
            print(f"Skipping filter {platform} — too few users remain.")

    if matching_users.empty:
        print("❌ No matching users found after filters.")
        return {"message": "No matching users found.", "skipped_filters": skipped_platforms}

    similar_user_ids = matching_users['user_id'].unique()
    df_similar_ratings = df_triple[df_triple['user_id'].isin(similar_user_ids)]

    recommendations = {}

    for genre in genre_list:
        real_genres = genre_aliases.get(genre, [genre])  # fallback to exact match

        # Step 3: Filter movies in that genre
        genre_movies = df_titles_genres[
            df_titles_genres['genres'].apply(lambda g: any(rg in g for rg in real_genres))
        ]
        genre_ratings = df_similar_ratings[
            df_similar_ratings['show_id'].isin(genre_movies.index)
        ]

        print(f"{genre} → {len(genre_movies)} movies, {len(genre_ratings)} ratings by similar users")

        if genre_ratings.empty:
            continue

        # Step 4: Find top-rated movies in that genre
        top_movies = (
            genre_ratings.groupby('show_id')['rating']
            .mean()
            .sort_values(ascending=False)
            .head(3)
            .index
            .tolist()
        )

        if not top_movies:
            continue

        # Step 5: Use the top movie as a seed for recommendations
        seed_id = top_movies[0]
        rec_ids, _ = recommend(seed_id, X, item_mapper, item_inv_mapper, k=k, messages=False)

        # Step 6: Filter recommendations to include only those in the same genre
        filtered_titles = [
            df_titles_genres.loc[i, 'title']
            for i in rec_ids
            if i in df_titles_genres.index and any(rg in df_titles_genres.loc[i, 'genres'] for rg in real_genres)
        ][:max_recs]

        if filtered_titles:
            recommendations[genre] = filtered_titles

    # Final return
    if not recommendations:
        return {"message": "No genre-based recommendations found.", "skipped_filters": skipped_platforms}

    return {
        "recommendations": recommendations,
        "skipped_filters": skipped_platforms
    }


def recommend(itemId, X, item_mapper, item_inv_mapper, k, metric='cosine', messages=True):
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    rec_ids = []
    item = item_mapper[itemId]
    item_vector = X[item]
    knn = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric).fit(X)
    rec = knn.kneighbors(item_vector.reshape(1, -1), return_distance=True)
    rec_indeces = rec[1][0]
    rec_distances = rec[0][0]
    rec_distances = np.delete(rec_distances, 0)

    for i in range(1, knn.n_neighbors):
        rec_ids.append(item_inv_mapper[rec_indeces[i]])

    return rec_ids, rec_distances

