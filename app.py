from flask import Flask, request, jsonify
import pandas as pd
import ast
from recommender import get_genre_recommendations_for_user, recommend, load_data
from flask_cors import CORS


# --- Load Data ---
df_users, df_triple, df_titles_genres, X, item_mapper, item_inv_mapper = load_data()

# --- Flask App ---
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"], supports_credentials=True)


@app.route('/recommendations', methods=['POST', 'OPTIONS'])
def recommend_for_new_user():
    if request.method == 'OPTIONS':
        return '', 200 
     
    try:
        data = request.get_json()

        age = data['age']
        gender = data['gender']
        platforms = data.get('platforms', {})  # optional
        genres = data.get('genres', ['Action', 'Romance', 'Comedy', 'Thrillers', 'Documentaries'])

        recs = get_genre_recommendations_for_user(
            age=age,
            gender=gender,
            platforms=platforms,
            genre_list=genres,
            df_users=df_users,
            df_triple=df_triple,
            df_titles_genres=df_titles_genres,
            X=X,
            item_mapper=item_mapper,
            item_inv_mapper=item_inv_mapper
        )

        return jsonify(recs)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5050)

