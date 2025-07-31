# collaborative_filtering.py - Complete Implementation
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix
from surprise import Dataset, Reader, SVD, NMF as SurpriseNMF, KNNBasic, accuracy
from surprise.model_selection import train_test_split as surprise_split
import pickle
import json
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class CollaborativeFilteringRecommender:
    def __init__(self, max_users=None, max_movies=None):
        """Initialize CF recommender with optional size limits for memory management"""
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.svd_model = None
        self.nmf_model = None
        self.surprise_svd = None
        self.surprise_nmf = None
        self.user_knn = None
        self.item_knn = None
        self.user_means = None
        self.global_mean = None
        self.max_users = max_users
        self.max_movies = max_movies
        self.svd_reconstructed = None
        self.nmf_reconstructed = None

    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        self.train_data = pd.read_csv('train_data.csv')
        self.val_data = pd.read_csv('validation_data.csv')
        self.test_data = pd.read_csv('test_data.csv')

        # Load metadata
        with open('dataset_metadata.json', 'r') as f:
            self.metadata = json.load(f)

        # Apply size limits if specified (for memory management)
        if self.max_users or self.max_movies:
            self.train_data = self._apply_size_limits(self.train_data)
            self.val_data = self._apply_size_limits(self.val_data)
            self.test_data = self._apply_size_limits(self.test_data)

        # Calculate basic statistics
        self.global_mean = self.train_data['rating'].mean()
        self.user_means = self.train_data.groupby('userId')['rating'].mean()

        print(f"Train data shape: {self.train_data.shape}")
        print(f"Validation data shape: {self.val_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        print(f"Global mean rating: {self.global_mean:.3f}")

    def _apply_size_limits(self, data):
        """Apply size limits to data for memory management"""
        if self.max_users:
            top_users = data['userId'].value_counts().head(self.max_users).index
            data = data[data['userId'].isin(top_users)]

        if self.max_movies:
            top_movies = data['movieId'].value_counts().head(self.max_movies).index
            data = data[data['movieId'].isin(top_movies)]

        return data

    def create_user_item_matrix(self, data=None):
        """Create user-item matrix"""
        if data is None:
            data = self.train_data

        print("Creating user-item matrix...")
        # Create pivot table
        self.user_item_matrix = data.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)

        print(f"User-item matrix shape: {self.user_item_matrix.shape}")

        # Create sparse matrix for memory efficiency
        self.user_item_sparse = csr_matrix(self.user_item_matrix.values)
        return self.user_item_matrix

    def user_based_collaborative_filtering(self, similarity_metric='cosine', n_neighbors=50):
        """Implement user-based collaborative filtering"""
        print(f"Computing user-based CF with {similarity_metric} similarity...")

        if self.user_item_matrix is None:
            self.create_user_item_matrix()

        # Center the ratings by subtracting user means
        user_matrix = self.user_item_matrix.values
        user_matrix_centered = user_matrix.copy()

        for i, user_id in enumerate(self.user_item_matrix.index):
            user_mean = self.user_means.get(user_id, self.global_mean)
            # Only center non-zero ratings
            mask = user_matrix[i] != 0
            user_matrix_centered[i, mask] = user_matrix[i, mask] - user_mean

        # Compute similarity matrix
        if similarity_metric == 'cosine':
            self.user_similarity = cosine_similarity(user_matrix_centered)
        elif similarity_metric == 'pearson':
            self.user_similarity = np.corrcoef(user_matrix_centered)
            self.user_similarity = np.nan_to_num(self.user_similarity)

        # Remove self-similarity
        np.fill_diagonal(self.user_similarity, 0)

        print("User similarity matrix computed!")
        return self.user_similarity

    def item_based_collaborative_filtering(self, similarity_metric='cosine', n_neighbors=50):
        """Implement item-based collaborative filtering"""
        print(f"Computing item-based CF with {similarity_metric} similarity...")

        if self.user_item_matrix is None:
            self.create_user_item_matrix()

        # Transpose for item-based
        item_matrix = self.user_item_matrix.T.values

        # Compute similarity matrix
        if similarity_metric == 'cosine':
            self.item_similarity = cosine_similarity(item_matrix)
        elif similarity_metric == 'pearson':
            self.item_similarity = np.corrcoef(item_matrix)
            self.item_similarity = np.nan_to_num(self.item_similarity)

        # Remove self-similarity
        np.fill_diagonal(self.item_similarity, 0)

        print("Item similarity matrix computed!")
        return self.item_similarity

    def matrix_factorization_sklearn(self, method='svd', n_components=50):
        """Implement matrix factorization using scikit-learn"""
        print(f"Training {method.upper()} model with {n_components} components...")

        if self.user_item_matrix is None:
            self.create_user_item_matrix()

        if method == 'svd':
            self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
            user_factors = self.svd_model.fit_transform(self.user_item_matrix)
            item_factors = self.svd_model.components_
            self.svd_reconstructed = np.dot(user_factors, item_factors)

        elif method == 'nmf':
            # NMF requires non-negative values
            nmf_matrix = self.user_item_matrix.copy()
            nmf_matrix = np.maximum(nmf_matrix, 0.1)  # Replace zeros with small positive values

            self.nmf_model = NMF(n_components=n_components, random_state=42, max_iter=200)
            user_factors = self.nmf_model.fit_transform(nmf_matrix)
            item_factors = self.nmf_model.components_
            self.nmf_reconstructed = np.dot(user_factors, item_factors)

        print(f"{method.upper()} model trained successfully!")
        return user_factors, item_factors

    def matrix_factorization_surprise(self, method='svd', n_factors=50, n_epochs=20):
        """Implement matrix factorization using Surprise library"""
        print(f"Training Surprise {method.upper()} model...")

        # Prepare data for Surprise
        reader = Reader(rating_scale=(0.5, 5.0))
        surprise_data = Dataset.load_from_df(
            self.train_data[['userId', 'movieId', 'rating']], reader
        )
        trainset = surprise_data.build_full_trainset()

        if method == 'svd':
            self.surprise_svd = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=42)
            self.surprise_svd.fit(trainset)
        elif method == 'nmf':
            self.surprise_nmf = SurpriseNMF(n_factors=n_factors, n_epochs=n_epochs, random_state=42)
            self.surprise_nmf.fit(trainset)

        print(f"Surprise {method.upper()} model trained successfully!")

    def train_knn_models(self, user_based=True, item_based=True):
        """Train KNN-based models using Surprise"""
        print("Training KNN models...")

        reader = Reader(rating_scale=(0.5, 5.0))
        surprise_data = Dataset.load_from_df(
            self.train_data[['userId', 'movieId', 'rating']], reader
        )
        trainset = surprise_data.build_full_trainset()

        if user_based:
            self.user_knn = KNNBasic(k=50, sim_options={'name': 'cosine', 'user_based': True})
            self.user_knn.fit(trainset)
            print("User-based KNN model trained!")

        if item_based:
            self.item_knn = KNNBasic(k=50, sim_options={'name': 'cosine', 'user_based': False})
            self.item_knn.fit(trainset)
            print("Item-based KNN model trained!")

    def predict_rating(self, user_id, item_id, method='svd'):
        """Predict rating for a user-item pair"""
        if method == 'user_based' and self.user_similarity is not None:
            return self._predict_user_based(user_id, item_id)
        elif method == 'item_based' and self.item_similarity is not None:
            return self._predict_item_based(user_id, item_id)
        elif method == 'svd':
            if self.surprise_svd:
                prediction = self.surprise_svd.predict(user_id, item_id)
                return prediction.est
            elif self.svd_reconstructed is not None:
                return self._predict_matrix_factorization(user_id, item_id, 'svd')
        elif method == 'nmf':
            if self.surprise_nmf:
                prediction = self.surprise_nmf.predict(user_id, item_id)
                return prediction.est
            elif self.nmf_reconstructed is not None:
                return self._predict_matrix_factorization(user_id, item_id, 'nmf')

        return self.global_mean  # Fallback

    def _predict_user_based(self, user_id, item_id):
        """Predict using user-based collaborative filtering"""
        if user_id not in self.user_item_matrix.index:
            return self.global_mean

        user_idx = self.user_item_matrix.index.get_loc(user_id)

        if item_id not in self.user_item_matrix.columns:
            return self.user_means.get(user_id, self.global_mean)

        item_idx = self.user_item_matrix.columns.get_loc(item_id)

        # Get similar users who rated this item
        item_ratings = self.user_item_matrix.iloc[:, item_idx]
        rated_users = item_ratings[item_ratings > 0].index

        if len(rated_users) == 0:
            return self.user_means.get(user_id, self.global_mean)

        # Calculate weighted average
        similarities = []
        ratings = []

        for rated_user in rated_users:
            if rated_user != user_id:
                rated_user_idx = self.user_item_matrix.index.get_loc(rated_user)
                sim = self.user_similarity[user_idx, rated_user_idx]
                if sim > 0:
                    similarities.append(sim)
                    rating = self.user_item_matrix.loc[rated_user, item_id]
                    user_mean = self.user_means.get(rated_user, self.global_mean)
                    ratings.append(rating - user_mean)

        if len(similarities) == 0:
            return self.user_means.get(user_id, self.global_mean)

        user_mean = self.user_means.get(user_id, self.global_mean)
        weighted_rating = np.average(ratings, weights=similarities)
        predicted_rating = user_mean + weighted_rating

        return np.clip(predicted_rating, 0.5, 5.0)

    def _predict_item_based(self, user_id, item_id):
        """Predict using item-based collaborative filtering"""
        if item_id not in self.user_item_matrix.columns:
            return self.global_mean

        if user_id not in self.user_item_matrix.index:
            item_ratings = self.user_item_matrix[item_id]
            return item_ratings[item_ratings > 0].mean() if len(
                item_ratings[item_ratings > 0]) > 0 else self.global_mean

        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index

        if len(rated_items) == 0:
            item_ratings = self.user_item_matrix[item_id]
            return item_ratings[item_ratings > 0].mean() if len(
                item_ratings[item_ratings > 0]) > 0 else self.global_mean

        # Calculate weighted average
        similarities = []
        ratings = []

        for rated_item in rated_items:
            if rated_item != item_id:
                rated_item_idx = self.user_item_matrix.columns.get_loc(rated_item)
                sim = self.item_similarity[item_idx, rated_item_idx]
                if sim > 0:
                    similarities.append(sim)
                    ratings.append(self.user_item_matrix.loc[user_id, rated_item])

        if len(similarities) == 0:
            item_ratings = self.user_item_matrix[item_id]
            return item_ratings[item_ratings > 0].mean() if len(
                item_ratings[item_ratings > 0]) > 0 else self.global_mean

        predicted_rating = np.average(ratings, weights=similarities)
        return np.clip(predicted_rating, 0.5, 5.0)

    def _predict_matrix_factorization(self, user_id, item_id, method='svd'):
        """Predict using matrix factorization"""
        if user_id not in self.user_item_matrix.index or item_id not in self.user_item_matrix.columns:
            return self.global_mean

        user_idx = self.user_item_matrix.index.get_loc(user_id)
        item_idx = self.user_item_matrix.columns.get_loc(item_id)

        if method == 'svd' and self.svd_reconstructed is not None:
            predicted_rating = self.svd_reconstructed[user_idx, item_idx]
        elif method == 'nmf' and self.nmf_reconstructed is not None:
            predicted_rating = self.nmf_reconstructed[user_idx, item_idx]
        else:
            return self.global_mean

        return np.clip(predicted_rating, 0.5, 5.0)

    def get_user_recommendations(self, user_id, method='svd', n_recommendations=10):
        """Get recommendations for a user"""
        if user_id not in self.user_item_matrix.index:
            # New user - recommend popular items
            return self._get_popular_recommendations(n_recommendations)

        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index

        if len(unrated_items) == 0:
            return []

        # Predict ratings for unrated items
        predictions = []
        for item_id in unrated_items:
            predicted_rating = self.predict_rating(user_id, item_id, method)
            predictions.append((item_id, predicted_rating))

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

    def _get_popular_recommendations(self, n_recommendations):
        """Get popular item recommendations"""
        item_popularity = self.train_data.groupby('movieId').agg({
            'rating': ['mean', 'count']
        })
        item_popularity.columns = ['avg_rating', 'count']

        # Filter items with at least 50 ratings
        popular_items = item_popularity[item_popularity['count'] >= 50]
        popular_items = popular_items.sort_values('avg_rating', ascending=False)

        recommendations = []
        for item_id in popular_items.head(n_recommendations).index:
            avg_rating = popular_items.loc[item_id, 'avg_rating']
            recommendations.append((item_id, avg_rating))

        return recommendations

    def evaluate_model(self, method='svd', metric='rmse'):
        """Evaluate model performance"""
        print(f"Evaluating {method} model using {metric}...")

        predictions = []
        actuals = []

        for _, row in self.val_data.iterrows():
            user_id = row['userId']
            item_id = row['movieId']
            actual_rating = row['rating']

            predicted_rating = self.predict_rating(user_id, item_id, method)

            predictions.append(predicted_rating)
            actuals.append(actual_rating)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        if metric == 'rmse':
            score = np.sqrt(mean_squared_error(actuals, predictions))
        elif metric == 'mae':
            score = mean_absolute_error(actuals, predictions)
        else:
            score = np.sqrt(mean_squared_error(actuals, predictions))

        print(f"{method.upper()} {metric.upper()}: {score:.4f}")
        return score

    def save_models(self, filename='collaborative_filtering_models.pkl'):
        """Save trained models"""
        models_data = {
            'user_similarity': self.user_similarity,
            'item_similarity': self.item_similarity,
            'svd_model': self.svd_model,
            'nmf_model': self.nmf_model,
            'surprise_svd': self.surprise_svd,
            'surprise_nmf': self.surprise_nmf,
            'user_knn': self.user_knn,
            'item_knn': self.item_knn,
            'svd_reconstructed': self.svd_reconstructed,
            'nmf_reconstructed': self.nmf_reconstructed,
            'user_item_matrix': self.user_item_matrix,
            'user_means': self.user_means,
            'global_mean': self.global_mean
        }

        with open(filename, 'wb') as f:
            pickle.dump(models_data, f)

        print(f"Models saved to {filename}")

    def train_all_models(self):
        """Train all collaborative filtering models"""
        print("Training all collaborative filtering models...")

        # Load data
        self.load_data()

        # Create user-item matrix
        self.create_user_item_matrix()

        # Train user-based CF
        self.user_based_collaborative_filtering()

        # Train item-based CF
        self.item_based_collaborative_filtering()

        # Train matrix factorization models
        self.matrix_factorization_sklearn('svd')
        self.matrix_factorization_sklearn('nmf')

        # Train Surprise models
        self.matrix_factorization_surprise('svd')
        self.matrix_factorization_surprise('nmf')

        # Train KNN models
        self.train_knn_models()

        print("All models trained successfully!")

        # Evaluate models
        print("\nEvaluating models...")
        methods = ['user_based', 'item_based', 'svd', 'nmf']
        for method in methods:
            try:
                self.evaluate_model(method)
            except Exception as e:
                print(f"Error evaluating {method}: {e}")

        # Save models
        self.save_models()

        return self


# Usage example for Team Member 2
if __name__ == "__main__":
    # Initialize recommender with size limits for memory management
    cf_recommender = CollaborativeFilteringRecommender(max_users=1000, max_movies=1000)

    # Train all models
    cf_recommender.train_all_models()

    # Example recommendations
    user_id = 1
    print(f"\nRecommendations for user {user_id}:")

    methods = ['svd', 'nmf', 'user_based', 'item_based']
    for method in methods:
        try:
            recommendations = cf_recommender.get_user_recommendations(user_id, method=method)
            print(f"\n{method.upper()} recommendations:")
            for item_id, score in recommendations[:5]:
                print(f"  Item {item_id}: {score:.3f}")
        except Exception as e:
            print(f"Error getting {method} recommendations: {e}")