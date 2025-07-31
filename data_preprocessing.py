# data_preprocessing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings('ignore')


class MovieLensDataProcessor:
    def __init__(self):
        self.ratings = None
        self.movies = None
        self.tags = None
        self.links = None
        self.merged_data = None

    def load_movielens_20m_data(self, data_path="movielens-20m-dataset"):
        """Load MovieLens 20M dataset from Kaggle"""
        try:
            print("Loading MovieLens 20M dataset...")

            # Load the main data files
            self.ratings = pd.read_csv(f'{data_path}/rating.csv')
            self.movies = pd.read_csv(f'{data_path}/movie.csv')
            self.tags = pd.read_csv(f'{data_path}/tag.csv') if os.path.exists(f'{data_path}/tag.csv') else None
            self.links = pd.read_csv(f'{data_path}/link.csv') if os.path.exists(f'{data_path}/link.csv') else None

            print("Data loaded successfully!")
            print(f"Ratings shape: {self.ratings.shape}")
            print(f"Movies shape: {self.movies.shape}")
            if self.tags is not None:
                print(f"Tags shape: {self.tags.shape}")
            if self.links is not None:
                print(f"Links shape: {self.links.shape}")

        except Exception as e:
            print(f"Error loading data: {e}")
            print("Creating sample dataset for demonstration...")
            self.create_sample_data()

    def create_sample_data(self):
        """Create a sample dataset for demonstration purposes"""
        np.random.seed(42)

        # Sample sizes for manageable computation
        n_users = 5000
        n_movies = 1000
        n_ratings = 50000

        print(f"Creating sample dataset with {n_users} users, {n_movies} movies, {n_ratings} ratings...")

        # Generate sample ratings
        user_ids = np.random.randint(1, n_users + 1, n_ratings)
        movie_ids = np.random.randint(1, n_movies + 1, n_ratings)
        ratings = np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                                   n_ratings, p=[0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20, 0.20, 0.10, 0.05])
        timestamps = np.random.randint(946684800, 1577836800, n_ratings)  # 2000-2020

        self.ratings = pd.DataFrame({
            'userId': user_ids,
            'movieId': movie_ids,
            'rating': ratings,
            'timestamp': timestamps
        }).drop_duplicates(subset=['userId', 'movieId'])

        # Generate sample movies
        genres_list = [
            'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
            'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]

        movies_data = []
        for i in range(1, n_movies + 1):
            # Create movie title with year
            year = np.random.randint(1980, 2021)
            title = f"Movie_{i} ({year})"

            # Assign random genres
            n_genres = np.random.randint(1, 4)  # 1-3 genres per movie
            movie_genres = np.random.choice(genres_list, n_genres, replace=False)
            genres_str = '|'.join(movie_genres)

            movies_data.append({
                'movieId': i,
                'title': title,
                'genres': genres_str
            })

        self.movies = pd.DataFrame(movies_data)
        print("Sample data created successfully!")

    def clean_data(self):
        """Clean and preprocess the data"""
        print("Cleaning data...")

        # Remove duplicates
        initial_ratings = len(self.ratings)
        self.ratings = self.ratings.drop_duplicates()
        print(f"Removed {initial_ratings - len(self.ratings)} duplicate ratings")

        # Handle missing values
        self.ratings = self.ratings.dropna()
        self.movies = self.movies.dropna()

        # Ensure valid rating range (MovieLens 20M uses 0.5-5.0 scale)
        self.ratings = self.ratings[(self.ratings['rating'] >= 0.5) & (self.ratings['rating'] <= 5.0)]

        # Remove users with very few ratings (less than 10)
        user_counts = self.ratings['userId'].value_counts()
        active_users = user_counts[user_counts >= 10].index
        self.ratings = self.ratings[self.ratings['userId'].isin(active_users)]
        print(f"Kept {len(active_users)} users with >= 10 ratings")

        # Remove movies with very few ratings (less than 5)
        movie_counts = self.ratings['movieId'].value_counts()
        popular_movies = movie_counts[movie_counts >= 5].index
        self.ratings = self.ratings[self.ratings['movieId'].isin(popular_movies)]
        print(f"Kept {len(popular_movies)} movies with >= 5 ratings")

        # Filter movies to only those in ratings
        self.movies = self.movies[self.movies['movieId'].isin(self.ratings['movieId'])]

        print("Data cleaning completed!")
        print(f"Final ratings shape: {self.ratings.shape}")
        print(f"Final movies shape: {self.movies.shape}")

    def extract_features(self):
        """Extract additional features from the data"""
        print("Extracting features...")

        # Extract year from movie title
        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)$')
        self.movies['year'] = pd.to_numeric(self.movies['year'], errors='coerce')
        self.movies['year'] = self.movies['year'].fillna(self.movies['year'].median())

        # Create genre binary features
        all_genres = set()
        for genres_str in self.movies['genres'].dropna():
            if genres_str != '(no genres listed)':
                all_genres.update(genres_str.split('|'))

        self.all_genres = sorted(list(all_genres))

        # Create binary genre columns
        for genre in self.all_genres:
            self.movies[f'genre_{genre}'] = self.movies['genres'].str.contains(genre, na=False).astype(int)

        # Convert timestamp to datetime
        self.ratings['datetime'] = pd.to_datetime(self.ratings['timestamp'], unit='s')
        self.ratings['year'] = self.ratings['datetime'].dt.year
        self.ratings['month'] = self.ratings['datetime'].dt.month
        self.ratings['day_of_week'] = self.ratings['datetime'].dt.dayofweek

        print(f"Extracted {len(self.all_genres)} genre features")
        print("Feature extraction completed!")

    def perform_eda(self):
        """Perform Exploratory Data Analysis"""
        print("Performing Exploratory Data Analysis...")

        plt.figure(figsize=(20, 15))

        # 1. Rating distribution
        plt.subplot(3, 4, 1)
        rating_counts = self.ratings['rating'].value_counts().sort_index()
        plt.bar(rating_counts.index, rating_counts.values, color='skyblue', edgecolor='black')
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.xticks(rating_counts.index)

        # 2. User activity distribution
        plt.subplot(3, 4, 2)
        user_activity = self.ratings['userId'].value_counts()
        plt.hist(user_activity, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        plt.title('User Activity Distribution')
        plt.xlabel('Number of Ratings per User')
        plt.ylabel('Number of Users')
        plt.yscale('log')

        # 3. Movie popularity distribution
        plt.subplot(3, 4, 3)
        movie_popularity = self.ratings['movieId'].value_counts()
        plt.hist(movie_popularity, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
        plt.title('Movie Popularity Distribution')
        plt.xlabel('Number of Ratings per Movie')
        plt.ylabel('Number of Movies')
        plt.yscale('log')

        # 4. Average rating per user
        plt.subplot(3, 4, 4)
        avg_user_rating = self.ratings.groupby('userId')['rating'].mean()
        plt.hist(avg_user_rating, bins=30, color='gold', edgecolor='black', alpha=0.7)
        plt.title('Average Rating per User')
        plt.xlabel('Average Rating')
        plt.ylabel('Number of Users')

        # 5. Average rating per movie
        plt.subplot(3, 4, 5)
        avg_movie_rating = self.ratings.groupby('movieId')['rating'].mean()
        plt.hist(avg_movie_rating, bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
        plt.title('Average Rating per Movie')
        plt.xlabel('Average Rating')
        plt.ylabel('Number of Movies')

        # 6. Ratings over time
        plt.subplot(3, 4, 6)
        ratings_by_year = self.ratings.groupby('year').size()
        plt.plot(ratings_by_year.index, ratings_by_year.values, marker='o', color='red')
        plt.title('Ratings Over Time')
        plt.xlabel('Year')
        plt.ylabel('Number of Ratings')
        plt.xticks(rotation=45)

        # 7. Genre popularity
        plt.subplot(3, 4, 7)
        genre_popularity = {}
        for genre in self.all_genres[:10]:  # Top 10 genres
            genre_popularity[genre] = self.movies[f'genre_{genre}'].sum()

        genres = list(genre_popularity.keys())
        counts = list(genre_popularity.values())
        plt.barh(genres, counts, color='teal')
        plt.title('Top 10 Genres by Movie Count')
        plt.xlabel('Number of Movies')

        # 8. Movie release years
        plt.subplot(3, 4, 8)
        year_counts = self.movies['year'].value_counts().sort_index()
        plt.plot(year_counts.index, year_counts.values, color='orange')
        plt.title('Movies by Release Year')
        plt.xlabel('Release Year')
        plt.ylabel('Number of Movies')

        # 9. Rating distribution by day of week
        plt.subplot(3, 4, 9)
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ratings_by_day = self.ratings.groupby('day_of_week').size()
        plt.bar(range(7), [ratings_by_day.get(i, 0) for i in range(7)],
                color='lightblue', edgecolor='black')
        plt.title('Ratings by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Ratings')
        plt.xticks(range(7), day_names)

        # 10. Top rated movies
        plt.subplot(3, 4, 10)
        movie_stats = self.ratings.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).round(2)
        movie_stats.columns = ['avg_rating', 'rating_count']
        movie_stats = movie_stats[movie_stats['rating_count'] >= 50]  # At least 50 ratings
        top_movies = movie_stats.nlargest(10, 'avg_rating')

        # Get movie titles
        top_movie_titles = []
        for movie_id in top_movies.index:
            title = self.movies[self.movies['movieId'] == movie_id]['title'].iloc[0]
            short_title = title[:20] + '...' if len(title) > 20 else title
            top_movie_titles.append(short_title)

        plt.barh(range(len(top_movies)), top_movies['avg_rating'], color='gold')
        plt.title('Top 10 Highest Rated Movies (≥50 ratings)')
        plt.xlabel('Average Rating')
        plt.yticks(range(len(top_movies)), top_movie_titles)

        # 11. Rating variance analysis
        plt.subplot(3, 4, 11)
        movie_variance = self.ratings.groupby('movieId')['rating'].var()
        plt.hist(movie_variance.dropna(), bins=30, color='lightpink', edgecolor='black', alpha=0.7)
        plt.title('Rating Variance Distribution')
        plt.xlabel('Rating Variance')
        plt.ylabel('Number of Movies')

        # 12. User rating behavior
        plt.subplot(3, 4, 12)
        user_std = self.ratings.groupby('userId')['rating'].std()
        plt.hist(user_std.dropna(), bins=30, color='lightgray', edgecolor='black', alpha=0.7)
        plt.title('User Rating Std Distribution')
        plt.xlabel('Rating Standard Deviation')
        plt.ylabel('Number of Users')

        plt.tight_layout()
        plt.savefig('eda_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print comprehensive statistics
        self.print_data_statistics()

    def print_data_statistics(self):
        """Print comprehensive data statistics"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE DATA STATISTICS")
        print("=" * 60)

        # Basic statistics
        print(f"Total users: {self.ratings['userId'].nunique():,}")
        print(f"Total movies: {self.ratings['movieId'].nunique():,}")
        print(f"Total ratings: {len(self.ratings):,}")
        print(f"Rating scale: {self.ratings['rating'].min()} - {self.ratings['rating'].max()}")
        print(f"Average rating: {self.ratings['rating'].mean():.3f}")
        print(f"Rating standard deviation: {self.ratings['rating'].std():.3f}")

        # Sparsity calculation
        sparsity = 1 - (len(self.ratings) / (self.ratings['userId'].nunique() *
                                             self.ratings['movieId'].nunique()))
        print(f"Data sparsity: {sparsity:.6f} ({sparsity * 100:.4f}%)")

        # User statistics
        user_stats = self.ratings.groupby('userId').agg({
            'rating': ['count', 'mean', 'std']
        }).round(3)
        user_stats.columns = ['num_ratings', 'avg_rating', 'rating_std']

        print(f"\nUser Statistics:")
        print(f"Average ratings per user: {user_stats['num_ratings'].mean():.1f}")
        print(f"Median ratings per user: {user_stats['num_ratings'].median():.1f}")
        print(f"Max ratings by single user: {user_stats['num_ratings'].max():,}")
        print(f"Min ratings by user: {user_stats['num_ratings'].min()}")

        # Movie statistics
        movie_stats = self.ratings.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std']
        }).round(3)
        movie_stats.columns = ['num_ratings', 'avg_rating', 'rating_std']

        print(f"\nMovie Statistics:")
        print(f"Average ratings per movie: {movie_stats['num_ratings'].mean():.1f}")
        print(f"Median ratings per movie: {movie_stats['num_ratings'].median():.1f}")
        print(f"Max ratings for single movie: {movie_stats['num_ratings'].max():,}")
        print(f"Min ratings for movie: {movie_stats['num_ratings'].min()}")

        # Genre statistics
        print(f"\nGenre Statistics:")
        print(f"Total unique genres: {len(self.all_genres)}")
        genre_counts = {}
        for genre in self.all_genres:
            genre_counts[genre] = self.movies[f'genre_{genre}'].sum()

        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        print("Top 10 most common genres:")
        for i, (genre, count) in enumerate(sorted_genres[:10], 1):
            print(f"  {i:2d}. {genre}: {count:,} movies")

        # Temporal patterns
        year_range = self.movies['year']
        print(f"\nTemporal Patterns:")
        print(f"Movie years range: {year_range.min():.0f} - {year_range.max():.0f}")
        print(f"Rating years range: {self.ratings['year'].min()} - {self.ratings['year'].max()}")

        # Rating distribution
        print(f"\nRating Distribution:")
        rating_dist = self.ratings['rating'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            percentage = (count / len(self.ratings)) * 100
            print(f"  {rating}: {count:,} ({percentage:.1f}%)")

    def create_train_test_split(self, test_size=0.2, validation_size=0.1):
        """Create train-validation-test split with temporal considerations"""
        print(f"Creating train-validation-test split...")

        # Sort by timestamp for temporal split option
        self.ratings_sorted = self.ratings.sort_values('timestamp')

        # Method 1: Random split (stratified by rating)
        train_val, test = train_test_split(
            self.ratings,
            test_size=test_size,
            random_state=42,
            stratify=self.ratings['rating']
        )

        train, val = train_test_split(
            train_val,
            test_size=validation_size / (1 - test_size),  # Adjust for remaining data
            random_state=42,
            stratify=train_val['rating']
        )

        # Save datasets
        train.to_csv('train_data.csv', index=False)
        val.to_csv('validation_data.csv', index=False)
        test.to_csv('test_data.csv', index=False)
        self.movies.to_csv('movies_data.csv', index=False)

        print(f"Train data shape: {train.shape}")
        print(f"Validation data shape: {val.shape}")
        print(f"Test data shape: {test.shape}")
        print(f"Movies data shape: {self.movies.shape}")

        # Save additional metadata
        metadata = {
            'genres': self.all_genres,
            'n_users': self.ratings['userId'].nunique(),
            'n_movies': self.ratings['movieId'].nunique(),
            'rating_scale': [self.ratings['rating'].min(), self.ratings['rating'].max()],
            'sparsity': 1 - (len(self.ratings) / (self.ratings['userId'].nunique() *
                                                  self.ratings['movieId'].nunique()))
        }

        import json
        with open('dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return train, val, test


# Usage example
if __name__ == "__main__":
    processor = MovieLensDataProcessor()

    # Load data (you'll need to download from Kaggle first)
    processor.load_movielens_20m_data()

    # Clean and preprocess
    processor.clean_data()
    processor.extract_features()

    # Perform EDA
    processor.perform_eda()

    # Create splits
    train_data, val_data, test_data = processor.create_train_test_split()

    print("\nData preprocessing completed!")
    print("Files saved:")
    print("- train_data.csv")
    print("- validation_data.csv")
    print("- test_data.csv")
    print("- movies_data.csv")
    print("- dataset_metadata.json")
    print("- eda_plots.png")