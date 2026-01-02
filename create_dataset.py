import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Sample Netflix shows data
shows_data = [
    {"title": "Stranger Things", "genre": "Sci-Fi", "rating": 8.7, "duration": 51, "type": "TV Show"},
    {"title": "The Crown", "genre": "Drama", "rating": 8.6, "duration": 58, "type": "TV Show"},
    {"title": "Ozark", "genre": "Crime", "rating": 8.4, "duration": 60, "type": "TV Show"},
    {"title": "Black Mirror", "genre": "Sci-Fi", "rating": 8.8, "duration": 45, "type": "TV Show"},
    {"title": "The Witcher", "genre": "Fantasy", "rating": 8.2, "duration": 60, "type": "TV Show"},
    {"title": "Narcos", "genre": "Crime", "rating": 8.8, "duration": 49, "type": "TV Show"},
    {"title": "House of Cards", "genre": "Drama", "rating": 8.7, "duration": 51, "type": "TV Show"},
    {"title": "Orange Is the New Black", "genre": "Comedy", "rating": 8.1, "duration": 60, "type": "TV Show"},
    {"title": "Friends", "genre": "Comedy", "rating": 8.9, "duration": 22, "type": "TV Show"},
    {"title": "The Office", "genre": "Comedy", "rating": 8.9, "duration": 22, "type": "TV Show"},
    {"title": "Breaking Bad", "genre": "Crime", "rating": 9.4, "duration": 47, "type": "TV Show"},
    {"title": "Dark", "genre": "Sci-Fi", "rating": 8.8, "duration": 60, "type": "TV Show"},
    {"title": "Mindhunter", "genre": "Crime", "rating": 8.6, "duration": 54, "type": "TV Show"},
    {"title": "The Umbrella Academy", "genre": "Sci-Fi", "rating": 8.0, "duration": 50, "type": "TV Show"},
    {"title": "Bridgerton", "genre": "Romance", "rating": 7.3, "duration": 60, "type": "TV Show"},
    {"title": "Money Heist", "genre": "Crime", "rating": 8.3, "duration": 67, "type": "TV Show"},
    {"title": "Squid Game", "genre": "Thriller", "rating": 8.0, "duration": 56, "type": "TV Show"},
    {"title": "The Queen's Gambit", "genre": "Drama", "rating": 8.6, "duration": 60, "type": "TV Show"},
    {"title": "Lucifer", "genre": "Fantasy", "rating": 8.1, "duration": 42, "type": "TV Show"},
    {"title": "13 Reasons Why", "genre": "Drama", "rating": 7.6, "duration": 57, "type": "TV Show"},
    
    # Movies
    {"title": "Bird Box", "genre": "Thriller", "rating": 6.6, "duration": 124, "type": "Movie"},
    {"title": "The Irishman", "genre": "Crime", "rating": 7.8, "duration": 209, "type": "Movie"},
    {"title": "Roma", "genre": "Drama", "rating": 7.7, "duration": 135, "type": "Movie"},
    {"title": "Marriage Story", "genre": "Drama", "rating": 7.9, "duration": 137, "type": "Movie"},
    {"title": "The Platform", "genre": "Sci-Fi", "rating": 7.0, "duration": 94, "type": "Movie"},
    {"title": "Extraction", "genre": "Action", "rating": 6.7, "duration": 116, "type": "Movie"},
    {"title": "The Old Guard", "genre": "Action", "rating": 6.7, "duration": 125, "type": "Movie"},
    {"title": "Enola Holmes", "genre": "Mystery", "rating": 6.6, "duration": 123, "type": "Movie"},
    {"title": "The Trial of the Chicago 7", "genre": "Drama", "rating": 7.8, "duration": 129, "type": "Movie"},
    {"title": "I Care a Lot", "genre": "Thriller", "rating": 6.3, "duration": 118, "type": "Movie"},
    {"title": "Army of the Dead", "genre": "Horror", "rating": 5.8, "duration": 148, "type": "Movie"},
    {"title": "Red Notice", "genre": "Action", "rating": 6.3, "duration": 118, "type": "Movie"},
    {"title": "Don't Look Up", "genre": "Comedy", "rating": 7.2, "duration": 138, "type": "Movie"},
    {"title": "The Adam Project", "genre": "Sci-Fi", "rating": 6.7, "duration": 106, "type": "Movie"},
    {"title": "Spenser Confidential", "genre": "Action", "rating": 6.2, "duration": 111, "type": "Movie"},
    {"title": "6 Underground", "genre": "Action", "rating": 6.1, "duration": 128, "type": "Movie"},
    {"title": "The Kissing Booth", "genre": "Romance", "rating": 6.0, "duration": 105, "type": "Movie"},
    {"title": "To All the Boys I've Loved Before", "genre": "Romance", "rating": 7.0, "duration": 99, "type": "Movie"},
    {"title": "The Half of It", "genre": "Romance", "rating": 6.9, "duration": 104, "type": "Movie"},
    {"title": "Someone Great", "genre": "Romance", "rating": 6.2, "duration": 92, "type": "Movie"},
    
    # Documentaries
    {"title": "Making a Murderer", "genre": "Documentary", "rating": 8.6, "duration": 60, "type": "TV Show"},
    {"title": "Tiger King", "genre": "Documentary", "rating": 7.5, "duration": 47, "type": "TV Show"},
    {"title": "Wild Wild Country", "genre": "Documentary", "rating": 8.2, "duration": 65, "type": "TV Show"},
    {"title": "The Social Dilemma", "genre": "Documentary", "rating": 7.6, "duration": 94, "type": "Movie"},
    {"title": "My Octopus Teacher", "genre": "Documentary", "rating": 8.1, "duration": 85, "type": "Movie"},
    {"title": "American Factory", "genre": "Documentary", "rating": 7.4, "duration": 110, "type": "Movie"},
    {"title": "Icarus", "genre": "Documentary", "rating": 7.9, "duration": 121, "type": "Movie"},
    {"title": "Fyre: The Greatest Party That Never Happened", "genre": "Documentary", "rating": 7.2, "duration": 97, "type": "Movie"},
    
    # Additional shows for better clustering
    {"title": "Daredevil", "genre": "Action", "rating": 8.6, "duration": 54, "type": "TV Show"},
    {"title": "Jessica Jones", "genre": "Action", "rating": 7.9, "duration": 52, "type": "TV Show"},
    {"title": "The Punisher", "genre": "Action", "rating": 8.5, "duration": 53, "type": "TV Show"},
    {"title": "Luke Cage", "genre": "Action", "rating": 7.3, "duration": 55, "type": "TV Show"},
    {"title": "Iron Fist", "genre": "Action", "rating": 6.4, "duration": 55, "type": "TV Show"},
    {"title": "The Haunting of Hill House", "genre": "Horror", "rating": 8.6, "duration": 62, "type": "TV Show"},
    {"title": "The Haunting of Bly Manor", "genre": "Horror", "rating": 7.4, "duration": 54, "type": "TV Show"},
    {"title": "Russian Doll", "genre": "Comedy", "rating": 7.8, "duration": 28, "type": "TV Show"},
    {"title": "GLOW", "genre": "Comedy", "rating": 8.0, "duration": 34, "type": "TV Show"},
    {"title": "Big Mouth", "genre": "Comedy", "rating": 7.8, "duration": 27, "type": "TV Show"},
    {"title": "BoJack Horseman", "genre": "Comedy", "rating": 8.8, "duration": 25, "type": "TV Show"},
    {"title": "F is for Family", "genre": "Comedy", "rating": 8.0, "duration": 30, "type": "TV Show"},
]

# Create DataFrame
df = pd.DataFrame(shows_data)

# Add some additional features for clustering
df['year'] = np.random.randint(2015, 2024, size=len(df))
df['seasons'] = np.where(df['type'] == 'TV Show', np.random.randint(1, 6, size=len(df)), 1)
df['country'] = np.random.choice(['USA', 'UK', 'Spain', 'Germany', 'South Korea', 'Canada'], 
                                size=len(df), p=[0.5, 0.15, 0.1, 0.1, 0.1, 0.05])

# Save to CSV
df.to_csv('/Users/ceydaakin/netflix/netflix_shows.csv', index=False)

print(f"Dataset created successfully!")
print(f"Total shows: {len(df)}")
print(f"Genres: {df['genre'].unique()}")
print(f"Types: {df['type'].value_counts()}")
print("\nFirst few rows:")
print(df.head())