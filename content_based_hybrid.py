# content_based_hybrid.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import json
import re
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class ContentBasedHybridRecommender:
    def __init__(self):
        """Initialize Content-Based and Hybrid Recommender System"""
        self.movies_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.metadata = None

        # Content-based components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.genre_matrix = None
        self.year_features = None
        self.content_similarity_matrix = None

        # Hybrid components
        self.cf_recommender = None
        self.content_weights = {'genres': 0.4, 'year': 0.2, 'title': 0.4}
        self.hybrid_weights = {'collaborative': 0.6, 'content': 0.4}

        # Feature matrices
        self.combined_features = None
        self.scaler = StandardScaler()

        # Caching for performance
        self.similarity_cache = {}
        self.recommendation_cache = {}

    def load_data(self):
        """Load preprocessed data"""
        print("Loading data for content-based and hybrid recommendations...")

        try:
            self.movies_data = pd.read_csv('movies_data.csv')
            self.train_data = pd.read_csv('train_data.csv')
            self.val_data = pd.read_csv('validation_data.csv')
            self.test_data = pd.read_csv('test_data.csv')

            with open('dataset_metadata.json', 'r') as f:
                self.metadata = json.load(f)

            print(f"Movies data shape: {self.movies_data.shape}")
            print(f"Train data shape: {self.train_data.shape}")

        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please run data_preprocessing.py first!")
            return False

        return True

    def preprocess_content_features(self):
        """Extract and preprocess content features from movies"""
        print("Preprocessing content features...")

        # Clean and prepare movie titles for text analysis
        self.movies_data['clean_title'] = self.movies_data['title'].apply(self._clean_title)

        # Extract genres into list format
        self.movies_data['genre_list'] = self.movies_data['genres'].apply(
            lambda x: x.split('|') if pd.notna(x) and x != '(no genres listed)' else []
        )

        # Create genre binary matrix
        mlb = MultiLabelBinarizer()
        genre_binary = mlb.fit_transform(self.movies_data['genre_list'])
        self.genre_matrix = pd.DataFrame(
            genre_binary,
            columns=mlb.classes_,
            index=self.movies_data['movieId']
        )

        # Normalize year features
        year_mean = self.movies_data['year'].mean()
        year_std = self.movies_data['year'].std()
        self.movies_data['year_normalized'] = (self.movies_data['year'] - year_mean) / year_std

        # Create TF-IDF features for movie titles
        self._create_tfidf_features()

        # Combine all content features
        self._combine_content_features()

        print("Content features preprocessing completed!")

    def _clean_title(self, title):
        """Clean movie title for text analysis"""
        if pd.isna(title):
            return ""

        # Remove year from title
        title_clean = re.sub(r'\(\d{4}\)', '', title)

        # Remove special characters and normalize
        title_clean = re.sub(r'[^\w\s]', ' ', title_clean)
        title_clean = ' '.join(title_clean.split())

        return title_clean.lower()

    def _create_tfidf_features(self):
        """Create TF-IDF features from movie titles and genres"""
        # Combine title and genres for richer text representation
        text_features = []

        for _, movie in self.movies_data.iterrows():
            text_content = movie['clean_title']

            # Add genres as text
            if movie['genre_list']:
                genre_text = ' '.join([g.lower().replace('-', ' ') for g in movie['genre_list']])
                text_content += ' ' + genre_text

            text_features.append(text_content)

        # Create TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    def _combine_content_features(self):
        """Combine all content features into a single matrix"""
        print("Combining content features...")

        # Get TF-IDF features as dense array
        tfidf_dense = self.tfidf_matrix.toarray()

        # Get genre features
        genre_features = self.genre_matrix.values

        # Get year features
        year_features = self.movies_data['year_normalized'].values.reshape(-1, 1)

        # Combine features with weights
        weighted_tfidf = tfidf_dense * self.content_weights['title']
        weighted_genres = genre_features * self.content_weights['genres']
        weighted_year = year_features * self.content_weights['year']

        # Concatenate all features
        self.combined_features = np.hstack([
            weighted_tfidf,
            weighted_genres,
            weighted_year
        ])

        # Normalize combined features
        self.combined_features = self.scaler.fit_transform(self.combined_features)

        print(f"Combined features shape: {self.combined_features.shape}")

    def compute_content_similarity(self):
        """Compute content-based similarity matrix"""
        print("Computing content-based similarity matrix...")

        # Use cosine similarity for content features
        self.content_similarity_matrix = cosine_similarity(self.combined_features)

        # Remove self-similarity
        np.fill_diagonal(self.content_similarity_matrix, 0)

        print("Content similarity matrix computed!")

    def get_content_recommendations(self, movie_id, n_recommendations=10):
        """Get content-based recommendations for a movie"""
        if movie_id not in self.movies_data['movieId'].values:
            return []

        # Get movie index
        movie_idx = self.movies_data[self.movies_data['movieId'] == movie_id].index[0]

        # Get similarity scores
        sim_scores = list(enumerate(self.content_similarity_matrix[movie_idx]))

        # Sort by similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top similar movies (excluding the movie itself)
        similar_movies = sim_scores[1:n_recommendations + 1]

        # Return movie IDs and similarity scores
        recommendations = []
        for idx, score in similar_movies:
            similar_movie_id = self.movies_data.iloc[idx]['movieId']
            recommendations.append((similar_movie_id, score))

        return recommendations

    def get_user_content_recommendations(self, user_id, n_recommendations=10):
        """Get content-based recommendations for a user based on their rating history"""
        # Get user's rated movies
        user_ratings = self.train_data[self.train_data['userId'] == user_id]

        if len(user_ratings) == 0:
            return self._get_popular_content_recommendations(n_recommendations)

        # Get highly rated movies (rating >= 4.0)
        liked_movies = user_ratings[user_ratings['rating'] >= 4.0]['movieId'].tolist()

        if not liked_movies:
            liked_movies = user_ratings.nlargest(min(5, len(user_ratings)), 'rating')['movieId'].tolist()

        # Get content recommendations for each liked movie
        all_recommendations = defaultdict(float)

        for movie_id in liked_movies:
            movie_recs = self.get_content_recommendations(movie_id, n_recommendations * 2)

            for rec_movie_id, score in movie_recs:
                # Weight by user's rating of the source movie
                user_movie_rating = user_ratings[user_ratings['movieId'] == movie_id]['rating'].iloc[0]
                weighted_score = score * (user_movie_rating / 5.0)
                all_recommendations[rec_movie_id] += weighted_score

        # Remove movies user has already rated
        rated_movies = set(user_ratings['movieId'].tolist())
        filtered_recommendations = {
            movie_id: score for movie_id, score in all_recommendations.items()
            if movie_id not in rated_movies
        }

        # Sort and return top recommendations
        sorted_recs = sorted(filtered_recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n_recommendations]

    def _get_popular_content_recommendations(self, n_recommendations):
        """Get popular movie recommendations for new users"""
        # Calculate movie popularity and average rating
        movie_stats = self.train_data.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).round(3)
        movie_stats.columns = ['avg_rating', 'rating_count']

        # Filter movies with sufficient ratings
        popular_movies = movie_stats[movie_stats['rating_count'] >= 50]

        # Sort by a combination of rating and popularity
        popular_movies['popularity_score'] = (
                popular_movies['avg_rating'] * 0.7 +
                np.log(popular_movies['rating_count']) * 0.3
        )

        top_movies = popular_movies.nlargest(n_recommendations, 'popularity_score')

        recommendations = []
        for movie_id in top_movies.index:
            score = top_movies.loc[movie_id, 'popularity_score']
            recommendations.append((movie_id, score))

        return recommendations

    def load_collaborative_model(self, cf_recommender=None):
        """Load collaborative filtering model for hybrid recommendations"""
        if cf_recommender is not None:
            self.cf_recommender = cf_recommender
            print("Collaborative filtering model loaded for hybrid recommendations!")
        else:
            try:
                # Try to load from pickle file
                with open('collaborative_filtering_models.pkl', 'rb') as f:
                    cf_data = pickle.load(f)

                print("Collaborative filtering model loaded from file!")
                return cf_data
            except FileNotFoundError:
                print("Collaborative filtering model not found. Hybrid recommendations will use content-based only.")
                return None

    def get_hybrid_recommendations(self, user_id, n_recommendations=10, method='weighted'):
        """Get hybrid recommendations combining collaborative and content-based"""

        # Get content-based recommendations
        content_recs = self.get_user_content_recommendations(user_id, n_recommendations * 2)
        content_dict = dict(content_recs)

        # Get collaborative filtering recommendations if available
        cf_recs = {}
        if self.cf_recommender is not None:
            try:
                cf_recommendations = self.cf_recommender.get_user_recommendations(
                    user_id, method='svd', n_recommendations=n_recommendations * 2
                )
                cf_recs = dict(cf_recommendations)
            except Exception as e:
                print(f"Error getting CF recommendations: {e}")

        # Combine recommendations based on method
        if method == 'weighted':
            return self._weighted_hybrid(content_dict, cf_recs, n_recommendations)
        elif method == 'switching':
            return self._switching_hybrid(user_id, content_dict, cf_recs, n_recommendations)
        elif method == 'mixed':
            return self._mixed_hybrid(content_dict, cf_recs, n_recommendations)
        else:
            return self._weighted_hybrid(content_dict, cf_recs, n_recommendations)

    def _weighted_hybrid(self, content_recs, cf_recs, n_recommendations):
        """Weighted hybrid combining both recommendation types"""
        combined_scores = defaultdict(float)

        # Add content-based recommendations
        for movie_id, score in content_recs.items():
            combined_scores[movie_id] += score * self.hybrid_weights['content']

        # Add collaborative filtering recommendations
        for movie_id, score in cf_recs.items():
            combined_scores[movie_id] += score * self.hybrid_weights['collaborative']

        # Sort and return top recommendations
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n_recommendations]

    def _switching_hybrid(self, user_id, content_recs, cf_recs, n_recommendations):
        """Switching hybrid based on user profile richness"""
        user_rating_count = len(self.train_data[self.train_data['userId'] == user_id])

        # Use collaborative filtering for users with many ratings, content-based for sparse users
        if user_rating_count >= 20 and cf_recs:
            return list(cf_recs.items())[:n_recommendations]
        else:
            return list(content_recs.items())[:n_recommendations]

    def _mixed_hybrid(self, content_recs, cf_recs, n_recommendations):
        """Mixed hybrid alternating between recommendation types"""
        mixed_recs = []
        content_list = list(content_recs.items())
        cf_list = list(cf_recs.items())

        # Alternate between content and collaborative recommendations
        i = j = 0
        while len(mixed_recs) < n_recommendations and (i < len(content_list) or j < len(cf_list)):
            # Add content recommendation
            if i < len(content_list):
                mixed_recs.append(content_list[i])
                i += 1

            # Add collaborative recommendation
            if len(mixed_recs) < n_recommendations and j < len(cf_list):
                cf_rec = cf_list[j]
                # Avoid duplicates
                if cf_rec not in mixed_recs:
                    mixed_recs.append(cf_rec)
                j += 1

        return mixed_recs

    def explain_recommendation(self, user_id, movie_id):
        """Provide explanation for why a movie was recommended"""
        explanations = []

        # Get movie details
        movie_info = self.movies_data[self.movies_data['movieId'] == movie_id]
        if movie_info.empty:
            return "Movie not found in database."

        movie_title = movie_info.iloc[0]['title']
        movie_genres = movie_info.iloc[0]['genres'].split('|')

        # Content-based explanations
        user_ratings = self.train_data[self.train_data['userId'] == user_id]
        if not user_ratings.empty:
            # Find user's favorite genres
            user_movies = user_ratings.merge(self.movies_data, on='movieId')
            user_high_rated = user_movies[user_movies['rating'] >= 4.0]

            if not user_high_rated.empty:
                user_genres = []
                for genres_str in user_high_rated['genres']:
                    user_genres.extend(genres_str.split('|'))

                from collections import Counter
                top_user_genres = Counter(user_genres).most_common(3)

                # Check genre overlap
                common_genres = set(movie_genres) & set([g[0] for g in top_user_genres])
                if common_genres:
                    explanations.append(f"You tend to enjoy {', '.join(common_genres)} movies")

                # Check for similar movies
                similar_movies = self.get_content_recommendations(movie_id, 5)
                user_rated_similar = []
                for sim_movie_id, _ in similar_movies:
                    user_rating = user_ratings[user_ratings['movieId'] == sim_movie_id]
                    if not user_rating.empty and user_rating.iloc[0]['rating'] >= 4.0:
                        similar_movie_title = self.movies_data[
                            self.movies_data['movieId'] == sim_movie_id
                            ].iloc[0]['title']
                        user_rated_similar.append(similar_movie_title)

                if user_rated_similar:
                    explanations.append(f"Similar to movies you liked: {', '.join(user_rated_similar[:2])}")

        if not explanations:
            explanations.append(f"Popular {', '.join(movie_genres[:2])} movie")

        return f"Recommended '{movie_title}': " + "; ".join(explanations)

    def evaluate_content_model(self, metric='precision_at_k', k=10):
        """Evaluate content-based model performance"""
        print(f"Evaluating content-based model using {metric}...")

        if metric == 'precision_at_k':
            return self._evaluate_precision_at_k(k)
        elif metric == 'diversity':
            return self._evaluate_diversity()
        elif metric == 'novelty':
            return self._evaluate_novelty()
        else:
            return self._evaluate_precision_at_k(k)

    def _evaluate_precision_at_k(self, k):
        """Evaluate precision@k for content-based recommendations"""
        precisions = []

        # Sample users for evaluation
        sample_users = self.val_data['userId'].unique()[:100]  # Evaluate on 100 users

        for user_id in sample_users:
            # Get user's validation ratings
            user_val_ratings = self.val_data[self.val_data['userId'] == user_id]
            liked_movies = set(user_val_ratings[user_val_ratings['rating'] >= 4.0]['movieId'])

            if len(liked_movies) > 0:
                # Get recommendations
                recommendations = self.get_user_content_recommendations(user_id, k)
                recommended_movies = set([movie_id for movie_id, _ in recommendations])

                # Calculate precision
                hits = len(recommended_movies & liked_movies)
                precision = hits / k if k > 0 else 0
                precisions.append(precision)

        avg_precision = np.mean(precisions) if precisions else 0
        print(f"Content-based Precision@{k}: {avg_precision:.4f}")
        return avg_precision

    def _evaluate_diversity(self):
        """Evaluate recommendation diversity"""
        # Get sample recommendations
        sample_users = self.train_data['userId'].unique()[:50]
        all_recommendations = []

        for user_id in sample_users:
            recs = self.get_user_content_recommendations(user_id, 10)
            all_recommendations.extend([movie_id for movie_id, _ in recs])

        # Calculate diversity as number of unique items / total recommendations
        unique_items = len(set(all_recommendations))
        total_recs = len(all_recommendations)
        diversity = unique_items / total_recs if total_recs > 0 else 0

        print(f"Content-based Diversity: {diversity:.4f}")
        return diversity

    def _evaluate_novelty(self):
        """Evaluate recommendation novelty"""
        # Calculate item popularity
        item_popularity = self.train_data['movieId'].value_counts()
        total_ratings = len(self.train_data)

        # Get sample recommendations
        sample_users = self.train_data['userId'].unique()[:50]
        novelty_scores = []

        for user_id in sample_users:
            recs = self.get_user_content_recommendations(user_id, 10)
            user_novelty = 0

            for movie_id, _ in recs:
                popularity = item_popularity.get(movie_id, 0)
                # Novelty is inverse of popularity
                novelty = -np.log2(popularity / total_ratings) if popularity > 0 else 10
                user_novelty += novelty

            if len(recs) > 0:
                novelty_scores.append(user_novelty / len(recs))

        avg_novelty = np.mean(novelty_scores) if novelty_scores else 0
        print(f"Content-based Novelty: {avg_novelty:.4f}")
        return avg_novelty

    def save_model(self, filename='content_based_hybrid_model.pkl'):
        """Save the content-based and hybrid model"""
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'genre_matrix': self.genre_matrix,
            'content_similarity_matrix': self.content_similarity_matrix,
            'combined_features': self.combined_features,
            'scaler': self.scaler,
            'content_weights': self.content_weights,
            'hybrid_weights': self.hybrid_weights,
            'movies_data': self.movies_data
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Content-based hybrid model saved to {filename}")

    def load_model(self, filename='content_based_hybrid_model.pkl'):
        """Load the content-based and hybrid model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)

            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.tfidf_matrix = model_data['tfidf_matrix']
            self.genre_matrix = model_data['genre_matrix']
            self.content_similarity_matrix = model_data['content_similarity_matrix']
            self.combined_features = model_data['combined_features']
            self.scaler = model_data['scaler']
            self.content_weights = model_data['content_weights']
            self.hybrid_weights = model_data['hybrid_weights']
            self.movies_data = model_data['movies_data']

            print(f"Content-based hybrid model loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Model file {filename} not found!")
            return False

    def train_model(self):
        """Train the complete content-based and hybrid model"""
        print("Training content-based and hybrid recommender system...")

        # Load data
        if not self.load_data():
            return False

        # Preprocess content features
        self.preprocess_content_features()

        # Compute content similarity
        self.compute_content_similarity()

        # Load collaborative filtering model if available
        self.load_collaborative_model()

        print("Content-based and hybrid model training completed!")

        # Evaluate model
        self.evaluate_content_model('precision_at_k', 10)
        self.evaluate_content_model('diversity')
        self.evaluate_content_model('novelty')

        # Save model
        self.save_model()

        return True


# Usage example
if __name__ == "__main__":
    # Initialize and train content-based hybrid recommender
    cb_hybrid = ContentBasedHybridRecommender()

    # Train the model
    if cb_hybrid.train_model():
        # Example recommendations
        user_id = 1
        print(f"\nContent-based recommendations for user {user_id}:")
        content_recs = cb_hybrid.get_user_content_recommendations(user_id, 5)
        for movie_id, score in content_recs:
            movie_title = cb_hybrid.movies_data[
                cb_hybrid.movies_data['movieId'] == movie_id
                ]['title'].iloc[0]
            print(f"  {movie_title}: {score:.3f}")

        # Example hybrid recommendations
        print(f"\nHybrid recommendations for user {user_id}:")
        hybrid_recs = cb_hybrid.get_hybrid_recommendations(user_id, 5)
        for movie_id, score in hybrid_recs:
            movie_title = cb_hybrid.movies_data[
                cb_hybrid.movies_data['movieId'] == movie_id
                ]['title'].iloc[0]
            print(f"  {movie_title}: {score:.3f}")

            # Show explanation
            explanation = cb_hybrid.explain_recommendation(user_id, movie_id)
            print(f"    Explanation: {explanation}")