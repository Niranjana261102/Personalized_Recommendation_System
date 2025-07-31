# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'movie_recommender_secret_key_2024'

# Global variables for models
cf_recommender = None
cb_hybrid_recommender = None
movies_data = None
metadata = None


def initialize_models():
    """Initialize and load all recommendation models"""
    global cf_recommender, cb_hybrid_recommender, movies_data, metadata
    logger.info("Initializing recommendation models...")

    try:
        # Load movies data
        if os.path.exists('movies_data.csv'):
            movies_data = pd.read_csv('movies_data.csv')
        else:
            logger.warning("movies_data.csv not found, creating dummy data")
            movies_data = create_dummy_movies_data()

        # Load metadata
        if os.path.exists('dataset_metadata.json'):
            with open('dataset_metadata.json', 'r') as f:
                metadata = json.load(f)
        else:
            logger.warning("dataset_metadata.json not found, creating default metadata")
            metadata = create_default_metadata()

        # Initialize models (placeholder - replace with actual model imports)
        # from collaborative_filtering import CollaborativeFilteringRecommender
        # from content_based_hybrid import ContentBasedHybridRecommender

        # For now, using placeholder classes
        cf_recommender = DummyCollaborativeRecommender()
        cb_hybrid_recommender = DummyHybridRecommender()

        logger.info("All models initialized successfully!")
        return True

    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        return False


def create_dummy_movies_data():
    """Create dummy movie data for testing"""
    return pd.DataFrame({
        'movieId': range(1, 101),
        'title': [f'Movie {i}' for i in range(1, 101)],
        'genres': ['Action|Adventure'] * 100,
        'year': [2020 + (i % 5) for i in range(100)]
    })


def create_default_metadata():
    """Create default metadata"""
    return {
        'total_users': 1000,
        'total_ratings': 50000,
        'avg_rating': 3.5,
        'dataset_name': 'Movie Recommendation Dataset'
    }


class DummyCollaborativeRecommender:
    """Placeholder for collaborative filtering recommender"""

    def __init__(self):
        self.user_item_matrix = None
        self.surprise_svd = None
        self.user_means = None
        self.global_mean = 3.5

    def get_recommendations(self, user_id, n_recommendations=10):
        """Return dummy recommendations"""
        return [(i, 4.0 + np.random.random()) for i in range(1, n_recommendations + 1)]

    def add_rating(self, user_id, movie_id, rating):
        """Add a new rating (placeholder)"""
        pass


class DummyHybridRecommender:
    """Placeholder for hybrid recommender"""

    def __init__(self):
        self.collaborative_model = None

    def load_model(self):
        return False

    def load_collaborative_model(self, cf_model):
        self.collaborative_model = cf_model

    def get_content_recommendations(self, user_id, n_recommendations=10):
        return [(i, 4.0 + np.random.random()) for i in range(1, n_recommendations + 1)]

    def get_hybrid_recommendations(self, user_id, n_recommendations=10):
        return [(i, 4.0 + np.random.random()) for i in range(1, n_recommendations + 1)]

    def get_similar_movies(self, movie_id, n_recommendations=10):
        return [(i, 0.8 + np.random.random() * 0.2) for i in range(1, n_recommendations + 1)]


@app.route('/')
def index():
    """Home page with movie search and user selection"""
    try:
        # Get sample of popular movies for display
        if movies_data is not None:
            sample_movies = movies_data.sample(min(20, len(movies_data))).to_dict('records')
        else:
            sample_movies = []

        return render_template('index.html',
                               movies=sample_movies,
                               total_movies=len(movies_data) if movies_data is not None else 0)
    except Exception as e:
        logger.error(f"Error in home route: {e}")
        return render_template('error.html', error="Error loading home page")


@app.route('/analytics')
def analytics():
    """Analytics page"""
    try:
        stats = {
            'total_movies': len(movies_data) if movies_data is not None else 0,
            'total_users': metadata.get('total_users', 0) if metadata else 0,
            'total_ratings': len(session.get('ratings', {})),
            'avg_rating': metadata.get('avg_rating', 0) if metadata else 0
        }
        return render_template('analytics.html', stats=stats)
    except Exception as e:
        logger.error(f"Error in analytics route: {e}")
        return render_template('error.html', error="Error loading analytics page")


@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """Handle recommendation form submission"""
    try:
        user_id = int(request.form.get('user_id'))
        algorithm = request.form.get('algorithm')
        num_recommendations = int(request.form.get('num_recommendations', 10))

        recommendations = []

        if algorithm == 'collaborative' and cf_recommender:
            recs = cf_recommender.get_recommendations(user_id, n_recommendations=num_recommendations)
            if recs:
                for movie_id, score in recs:
                    movie_info = get_movie_info(movie_id)
                    if movie_info:
                        movie_info['rating'] = score
                        recommendations.append(movie_info)

        elif algorithm == 'content' and cb_hybrid_recommender:
            recs = cb_hybrid_recommender.get_content_recommendations(user_id, n_recommendations=num_recommendations)
            if recs:
                for movie_id, score in recs:
                    movie_info = get_movie_info(movie_id)
                    if movie_info:
                        movie_info['similarity_score'] = score
                        recommendations.append(movie_info)

        elif algorithm == 'hybrid' and cb_hybrid_recommender:
            recs = cb_hybrid_recommender.get_hybrid_recommendations(user_id, n_recommendations=num_recommendations)
            if recs:
                for movie_id, score in recs:
                    movie_info = get_movie_info(movie_id)
                    if movie_info:
                        movie_info['rating'] = score
                        recommendations.append(movie_info)

        return render_template('recommendations.html',
                               recommendations=recommendations,
                               algorithm=algorithm,
                               user_id=user_id)

    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return render_template('error.html', error="Failed to get recommendations")


@app.route('/api/search_movies')
def search_movies():
    """API endpoint to search for movies"""
    try:
        query = request.args.get('q', '').lower()
        limit = int(request.args.get('limit', 10))

        if not query or movies_data is None:
            return jsonify([])

        # Search in movie titles
        filtered_movies = movies_data[
            movies_data['title'].str.lower().str.contains(query, na=False)
        ].head(limit)

        results = []
        for _, movie in filtered_movies.iterrows():
            results.append({
                'id': int(movie['movieId']),
                'title': movie['title'],
                'genres': movie['genres'],
                'year': int(movie['year']) if pd.notna(movie['year']) else 'Unknown'
            })

        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in search_movies: {e}")
        return jsonify([])


@app.route('/api/get_recommendations')
def api_get_recommendations():
    """API endpoint to get recommendations for a user"""
    try:
        user_id = request.args.get('user_id', type=int)
        rec_type = request.args.get('type', 'hybrid')
        limit = int(request.args.get('limit', 10))

        if not user_id:
            return jsonify({'error': 'User ID is required'})

        recommendations = []

        if rec_type == 'collaborative' and cf_recommender:
            recs = cf_recommender.get_recommendations(user_id, n_recommendations=limit)
            if recs:
                for movie_id, score in recs:
                    movie_info = get_movie_info(movie_id)
                    if movie_info:
                        movie_info['score'] = float(score)
                        recommendations.append(movie_info)

        elif rec_type == 'content' and cb_hybrid_recommender:
            recs = cb_hybrid_recommender.get_content_recommendations(user_id, n_recommendations=limit)
            if recs:
                for movie_id, score in recs:
                    movie_info = get_movie_info(movie_id)
                    if movie_info:
                        movie_info['score'] = float(score)
                        recommendations.append(movie_info)

        elif rec_type == 'hybrid' and cb_hybrid_recommender:
            recs = cb_hybrid_recommender.get_hybrid_recommendations(user_id, n_recommendations=limit)
            if recs:
                for movie_id, score in recs:
                    movie_info = get_movie_info(movie_id)
                    if movie_info:
                        movie_info['score'] = float(score)
                        recommendations.append(movie_info)

        return jsonify({
            'recommendations': recommendations,
            'type': rec_type,
            'user_id': user_id
        })
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return jsonify({'error': 'Failed to get recommendations'})


@app.route('/api/rate_movie', methods=['POST'])
def rate_movie():
    """API endpoint to rate a movie"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        movie_id = data.get('movie_id')
        rating = data.get('rating')

        if not all([user_id, movie_id, rating]):
            return jsonify({'error': 'Missing required parameters'})

        # Store rating in session for demo purposes
        if 'ratings' not in session:
            session['ratings'] = {}
        session['ratings'][f"{user_id}_{movie_id}"] = {
            'rating': rating,
            'timestamp': datetime.now().isoformat()
        }

        # Update models with new rating
        if cf_recommender:
            cf_recommender.add_rating(user_id, movie_id, rating)

        return jsonify({'success': True, 'message': 'Rating saved successfully'})
    except Exception as e:
        logger.error(f"Error rating movie: {e}")
        return jsonify({'error': 'Failed to save rating'})


@app.route('/api/user_ratings/<int:user_id>')
def get_user_ratings(user_id):
    """Get all ratings for a specific user"""
    try:
        user_ratings = []
        # Get ratings from session
        if 'ratings' in session:
            for key, rating_data in session['ratings'].items():
                if key.startswith(f"{user_id}_"):
                    movie_id = int(key.split('_')[1])
                    movie_info = get_movie_info(movie_id)
                    if movie_info:
                        movie_info.update({
                            'user_rating': rating_data['rating'],
                            'rated_at': rating_data['timestamp']
                        })
                        user_ratings.append(movie_info)

        return jsonify({
            'user_id': user_id,
            'ratings': user_ratings,
            'total_ratings': len(user_ratings)
        })
    except Exception as e:
        logger.error(f"Error getting user ratings: {e}")
        return jsonify({'error': 'Failed to get user ratings'})


@app.route('/user/<int:user_id>')
def user_dashboard(user_id):
    """User dashboard with recommendations and ratings"""
    try:
        return render_template('user_dashboard.html', user_id=user_id)
    except Exception as e:
        logger.error(f"Error in user dashboard: {e}")
        return render_template('error.html', error="Error loading user dashboard")


@app.route('/api/movie/<int:movie_id>')
def get_movie_details(movie_id):
    """Get detailed information about a specific movie"""
    try:
        movie_info = get_movie_info(movie_id)
        if movie_info:
            return jsonify(movie_info)
        else:
            return jsonify({'error': 'Movie not found'}), 404
    except Exception as e:
        logger.error(f"Error getting movie details: {e}")
        return jsonify({'error': 'Failed to get movie details'}), 500


@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    try:
        stats = {
            'total_movies': len(movies_data) if movies_data is not None else 0,
            'total_ratings': len(session.get('ratings', {})),
            'models_loaded': {
                'collaborative_filtering': cf_recommender is not None,
                'content_based_hybrid': cb_hybrid_recommender is not None
            }
        }
        if metadata:
            stats.update(metadata)

        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Failed to get statistics'})


@app.route('/api/similar_movies/<int:movie_id>')
def get_similar_movies(movie_id):
    """Get movies similar to a given movie"""
    try:
        limit = int(request.args.get('limit', 10))
        similar_movies = []

        if cb_hybrid_recommender and hasattr(cb_hybrid_recommender, 'get_similar_movies'):
            similar = cb_hybrid_recommender.get_similar_movies(movie_id, n_recommendations=limit)
            if similar:
                for sim_movie_id, score in similar:
                    movie_info = get_movie_info(sim_movie_id)
                    if movie_info:
                        movie_info['similarity_score'] = float(score)
                        similar_movies.append(movie_info)

        return jsonify({
            'movie_id': movie_id,
            'similar_movies': similar_movies
        })
    except Exception as e:
        logger.error(f"Error getting similar movies: {e}")
        return jsonify({'error': 'Failed to get similar movies'})


def get_movie_info(movie_id):
    """Helper function to get movie information by ID"""
    try:
        if movies_data is None:
            return None

        movie = movies_data[movies_data['movieId'] == movie_id]
        if movie.empty:
            return None

        movie_row = movie.iloc[0]
        return {
            'id': int(movie_row['movieId']),
            'title': movie_row['title'],
            'genres': movie_row['genres'],
            'year': int(movie_row['year']) if pd.notna(movie_row['year']) else 'Unknown'
        }
    except Exception as e:
        logger.error(f"Error getting movie info for ID {movie_id}: {e}")
        return None


@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500


def create_app():
    """Application factory function"""
    # Initialize models when creating the app
    if not initialize_models():
        logger.error("Failed to initialize models!")
        raise RuntimeError("Failed to initialize recommendation models")
    return app


if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)

    # Initialize models
    if initialize_models():
        logger.info("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to initialize models. Exiting...")
        exit(1)