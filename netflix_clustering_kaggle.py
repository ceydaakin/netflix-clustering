"""
Netflix Show Clustering - Real Kaggle Dataset Version
=====================================================

This script analyzes the real Netflix dataset from Kaggle:
https://www.kaggle.com/datasets/shivamb/netflix-shows

Dataset: netflix_titles.csv (8,807 shows)
Tech Stack: Python, Pandas, Scikit-learn, Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
import os
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NetflixKaggleAnalysis:
    def __init__(self, data_path='netflix_titles.csv'):
        """Initialize Netflix Kaggle dataset analysis"""
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.kmeans = None
        self.optimal_clusters = None
        
    def check_dataset(self):
        """Check if dataset exists"""
        if not os.path.exists(self.data_path):
            print("❌ Dataset not found!")
            print("\n📥 Please download from:")
            print("   https://www.kaggle.com/datasets/shivamb/netflix-shows")
            return False
        return True
    
    def load_data(self):
        """Load and display basic information"""
        if not self.check_dataset():
            return None
            
        print("🎬 Loading Netflix Kaggle Dataset...")
        print("=" * 70)
        
        self.df = pd.read_csv(self.data_path)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"\n📊 Dataset Info:")
        print(f"   Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"   Columns: {list(self.df.columns)}")
        
        print(f"\n🔍 Missing Values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0])
        
        print(f"\n👀 Sample Data:")
        print(self.df[['title', 'type', 'release_year', 'rating', 'duration', 'listed_in']].head())
        
        return self
    
    def preprocess_data(self):
        """Preprocess Kaggle dataset for clustering"""
        if self.df is None:
            print("❌ Please load data first!")
            return None
            
        print("\n🔧 PREPROCESSING DATA")
        print("=" * 70)
        
        self.df_processed = self.df.copy()
        
        # Extract year from date_added
        print("📅 Extracting year from date_added...")
        self.df_processed['year_added'] = pd.to_datetime(
            self.df_processed['date_added'], errors='coerce'
        ).dt.year
        
        # Process duration (convert to minutes)
        print("⏱️  Processing duration...")
        def extract_duration(duration_str, content_type):
            if pd.isna(duration_str):
                return np.nan
            try:
                if content_type == 'Movie':
                    # Extract minutes from "XX min"
                    return int(duration_str.split()[0])
                else:
                    # For TV Shows, use number of seasons * 10 episodes * 45 min average
                    seasons = int(duration_str.split()[0])
                    return seasons * 10 * 45  # Approximate total minutes
            except:
                return np.nan
        
        self.df_processed['duration_minutes'] = self.df_processed.apply(
            lambda x: extract_duration(x['duration'], x['type']), axis=1
        )
        
        # Extract primary genre from listed_in
        print("🎭 Extracting primary genre...")
        self.df_processed['primary_genre'] = self.df_processed['listed_in'].apply(
            lambda x: x.split(',')[0].strip() if pd.notna(x) else 'Unknown'
        )
        
        # Extract primary country
        print("🌍 Extracting primary country...")
        self.df_processed['primary_country'] = self.df_processed['country'].apply(
            lambda x: x.split(',')[0].strip() if pd.notna(x) else 'Unknown'
        )
        
        # Create rating score (map rating categories to numbers)
        print("⭐ Creating rating score...")
        rating_map = {
            'G': 1, 'TV-Y': 1, 'TV-G': 1,
            'PG': 2, 'TV-Y7': 2, 'TV-Y7-FV': 2,
            'PG-13': 3, 'TV-PG': 3,
            'R': 4, 'TV-14': 4,
            'NC-17': 5, 'TV-MA': 5,
            'NR': 3, 'UR': 3  # Unrated as middle
        }
        self.df_processed['rating_score'] = self.df_processed['rating'].map(
            lambda x: rating_map.get(x, 3)
        )
        
        # Handle missing values
        print("🧹 Handling missing values...")
        # Fill missing duration with median by type
        for content_type in ['Movie', 'TV Show']:
            mask = self.df_processed['type'] == content_type
            median_duration = self.df_processed.loc[mask, 'duration_minutes'].median()
            self.df_processed.loc[mask, 'duration_minutes'] = \
                self.df_processed.loc[mask, 'duration_minutes'].fillna(median_duration)
        
        # Fill missing year_added with release_year
        self.df_processed['year_added'] = self.df_processed['year_added'].fillna(
            self.df_processed['release_year']
        )
        
        # Encode categorical variables
        print("🔢 Encoding categorical variables...")
        categorical_cols = ['type', 'primary_genre', 'primary_country']
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df_processed[f'{col}_encoded'] = le.fit_transform(
                self.df_processed[col].fillna('Unknown')
            )
            self.label_encoders[col] = le
            print(f"   ✓ Encoded {col}: {len(le.classes_)} categories")
        
        # Select features for clustering
        clustering_features = [
            'release_year',
            'year_added',
            'duration_minutes',
            'rating_score',
            'type_encoded',
            'primary_genre_encoded',
            'primary_country_encoded'
        ]
        
        # Remove rows with missing values in clustering features
        self.df_processed = self.df_processed.dropna(subset=clustering_features)
        
        print(f"\n✅ Preprocessing complete!")
        print(f"   Features: {clustering_features}")
        print(f"   Final dataset size: {len(self.df_processed):,} shows")
        
        # Scale features
        print("\n📏 Scaling features...")
        X = self.df_processed[clustering_features]
        X_scaled = self.scaler.fit_transform(X)
        self.X_scaled = pd.DataFrame(X_scaled, columns=clustering_features)
        
        print(f"   ✓ Features scaled successfully")
        print(f"   Shape: {self.X_scaled.shape}")
        
        return self
    
    def explore_data(self):
        """Explore the Kaggle dataset"""
        if self.df_processed is None:
            print("❌ Please preprocess data first!")
            return None
            
        print("\n📊 EXPLORATORY DATA ANALYSIS")
        print("=" * 70)
        
        # Basic statistics
        print("\n📈 Content Statistics:")
        print(f"   Total Shows: {len(self.df_processed):,}")
        print(f"   Movies: {(self.df_processed['type'] == 'Movie').sum():,}")
        print(f"   TV Shows: {(self.df_processed['type'] == 'TV Show').sum():,}")
        
        print(f"\n🎭 Top 10 Genres:")
        print(self.df_processed['primary_genre'].value_counts().head(10))
        
        print(f"\n🌍 Top 10 Countries:")
        print(self.df_processed['primary_country'].value_counts().head(10))
        
        print(f"\n📅 Year Range:")
        print(f"   Release Year: {self.df_processed['release_year'].min()} - {self.df_processed['release_year'].max()}")
        print(f"   Year Added: {self.df_processed['year_added'].min():.0f} - {self.df_processed['year_added'].max():.0f}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🎬 Netflix Kaggle Dataset - Exploratory Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. Content type distribution
        self.df_processed['type'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['#E50914', '#221F1F'])
        axes[0, 0].set_title('Content Type Distribution')
        axes[0, 0].set_xlabel('Type')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=0)
        
        # 2. Top genres
        self.df_processed['primary_genre'].value_counts().head(10).plot(
            kind='barh', ax=axes[0, 1], color='coral'
        )
        axes[0, 1].set_title('Top 10 Genres')
        axes[0, 1].set_xlabel('Count')
        
        # 3. Content added over years
        year_counts = self.df_processed['year_added'].value_counts().sort_index()
        year_counts.plot(kind='line', ax=axes[0, 2], color='#E50914', linewidth=2)
        axes[0, 2].set_title('Content Added Over Years')
        axes[0, 2].set_xlabel('Year')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Duration distribution by type
        movies = self.df_processed[self.df_processed['type'] == 'Movie']['duration_minutes']
        tv_shows = self.df_processed[self.df_processed['type'] == 'TV Show']['duration_minutes']
        
        axes[1, 0].hist([movies, tv_shows], bins=30, label=['Movies', 'TV Shows'], 
                       color=['#E50914', '#221F1F'], alpha=0.7)
        axes[1, 0].set_title('Duration Distribution by Type')
        axes[1, 0].set_xlabel('Duration (minutes)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        
        # 5. Top countries
        self.df_processed['primary_country'].value_counts().head(10).plot(
            kind='barh', ax=axes[1, 1], color='skyblue'
        )
        axes[1, 1].set_title('Top 10 Countries')
        axes[1, 1].set_xlabel('Count')
        
        # 6. Rating distribution
        self.df_processed['rating'].value_counts().head(10).plot(
            kind='bar', ax=axes[1, 2], color='gold'
        )
        axes[1, 2].set_title('Top 10 Ratings')
        axes[1, 2].set_xlabel('Rating')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return self
    
    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters"""
        if self.X_scaled is None:
            print("❌ Please preprocess data first!")
            return None
            
        print("\n🔍 FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("=" * 70)
        
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        print("Testing different values of k...")
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
            
            sil_score = silhouette_score(self.X_scaled, kmeans.labels_)
            silhouette_scores.append(sil_score)
            print(f"   k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
        
        self.optimal_clusters = K_range[np.argmax(silhouette_scores)]
        print(f"\n✅ Optimal clusters: {self.optimal_clusters}")
        print(f"   Best Silhouette Score: {max(silhouette_scores):.3f}")
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=self.optimal_clusters, color='red', linestyle='--', 
                   label=f'Optimal k={self.optimal_clusters}')
        ax1.legend()
        
        ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=self.optimal_clusters, color='red', linestyle='--',
                   label=f'Optimal k={self.optimal_clusters}')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        return self
    
    def perform_clustering(self, n_clusters=None):
        """Perform K-Means clustering"""
        if n_clusters is None:
            n_clusters = self.optimal_clusters if self.optimal_clusters else 5
        
        print(f"\n🎯 PERFORMING K-MEANS CLUSTERING (k={n_clusters})")
        print("=" * 70)
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(self.X_scaled)
        
        self.df_processed['cluster'] = cluster_labels
        
        silhouette_avg = silhouette_score(self.X_scaled, cluster_labels)
        
        print(f"✅ Clustering complete!")
        print(f"   Inertia: {self.kmeans.inertia_:.2f}")
        print(f"   Silhouette Score: {silhouette_avg:.3f}")
        print(f"\n📊 Cluster Distribution:")
        print(self.df_processed['cluster'].value_counts().sort_index())
        
        return self
    
    def visualize_clusters(self):
        """Visualize clustering results"""
        if 'cluster' not in self.df_processed.columns:
            print("❌ Please perform clustering first!")
            return None
        
        print("\n🎨 VISUALIZING CLUSTERS")
        print("=" * 70)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. PCA scatter
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1],
                            c=self.df_processed['cluster'].values,
                            cmap='tab10', alpha=0.6, s=30)
        ax1.set_title('Clusters in PCA Space')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.colorbar(scatter, ax=ax1)
        
        # 2. Cluster sizes
        ax2 = plt.subplot(2, 3, 2)
        cluster_counts = self.df_processed['cluster'].value_counts().sort_index()
        bars = ax2.bar(cluster_counts.index, cluster_counts.values,
                      color=plt.cm.tab10(np.linspace(0, 1, len(cluster_counts))))
        ax2.set_title('Cluster Size Distribution')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Number of Shows')
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom')
        
        # 3. Type distribution by cluster
        ax3 = plt.subplot(2, 3, 3)
        type_cluster = pd.crosstab(self.df_processed['cluster'],
                                  self.df_processed['type'])
        type_cluster.plot(kind='bar', ax=ax3, color=['#E50914', '#221F1F'])
        ax3.set_title('Content Type by Cluster')
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Count')
        ax3.legend(title='Type')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)
        
        # 4. Genre distribution by cluster
        ax4 = plt.subplot(2, 3, 4)
        # Get top 5 genres
        top_genres = self.df_processed['primary_genre'].value_counts().head(5).index
        genre_cluster = pd.crosstab(
            self.df_processed['cluster'],
            self.df_processed[self.df_processed['primary_genre'].isin(top_genres)]['primary_genre']
        )
        genre_cluster.plot(kind='bar', stacked=True, ax=ax4, colormap='Set3')
        ax4.set_title('Top 5 Genres by Cluster')
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('Count')
        ax4.legend(title='Genre', bbox_to_anchor=(1.05, 1))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)
        
        # 5. Average duration by cluster
        ax5 = plt.subplot(2, 3, 5)
        avg_duration = self.df_processed.groupby('cluster')['duration_minutes'].mean()
        bars5 = ax5.bar(avg_duration.index, avg_duration.values,
                       color=plt.cm.tab10(np.linspace(0, 1, len(avg_duration))))
        ax5.set_title('Average Duration by Cluster')
        ax5.set_xlabel('Cluster')
        ax5.set_ylabel('Duration (minutes)')
        for bar in bars5:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom')
        
        # 6. Release year distribution
        ax6 = plt.subplot(2, 3, 6)
        for cluster in sorted(self.df_processed['cluster'].unique()):
            cluster_data = self.df_processed[self.df_processed['cluster'] == cluster]
            ax6.hist(cluster_data['release_year'], bins=20, alpha=0.5,
                    label=f'Cluster {cluster}')
        ax6.set_title('Release Year Distribution by Cluster')
        ax6.set_xlabel('Release Year')
        ax6.set_ylabel('Count')
        ax6.legend()
        
        plt.tight_layout()
        plt.show()
        
        return self
    
    def analyze_clusters(self):
        """Analyze each cluster in detail"""
        if 'cluster' not in self.df_processed.columns:
            print("❌ Please perform clustering first!")
            return None
        
        print("\n📊 DETAILED CLUSTER ANALYSIS")
        print("=" * 70)
        
        n_clusters = len(self.df_processed['cluster'].unique())
        
        for cluster_id in range(n_clusters):
            cluster_data = self.df_processed[self.df_processed['cluster'] == cluster_id]
            
            print(f"\n{'='*70}")
            print(f"CLUSTER {cluster_id}")
            print(f"{'='*70}")
            print(f"Size: {len(cluster_data):,} shows ({len(cluster_data)/len(self.df_processed)*100:.1f}%)")
            
            print(f"\n📊 Content Type:")
            print(cluster_data['type'].value_counts())
            
            print(f"\n🎭 Top 5 Genres:")
            print(cluster_data['primary_genre'].value_counts().head())
            
            print(f"\n🌍 Top 5 Countries:")
            print(cluster_data['primary_country'].value_counts().head())
            
            print(f"\n📅 Year Statistics:")
            print(f"   Release Year: {cluster_data['release_year'].min()} - {cluster_data['release_year'].max()}")
            print(f"   Average Release Year: {cluster_data['release_year'].mean():.0f}")
            
            print(f"\n⏱️  Duration Statistics:")
            print(f"   Average: {cluster_data['duration_minutes'].mean():.0f} minutes")
            print(f"   Median: {cluster_data['duration_minutes'].median():.0f} minutes")
            
            print(f"\n🎬 Sample Titles:")
            sample = cluster_data[['title', 'type', 'primary_genre', 'release_year']].head(5)
            for _, row in sample.iterrows():
                print(f"   • {row['title']} ({row['type']}, {row['primary_genre']}, {row['release_year']})")
        
        return self

def main():
    """Main function to run the complete analysis"""
    print("🎬 NETFLIX KAGGLE DATASET CLUSTERING ANALYSIS")
    print("=" * 70)
    print("Dataset: https://www.kaggle.com/datasets/shivamb/netflix-shows")
    print("Tech Stack: Python, Pandas, Scikit-learn, Seaborn")
    print("=" * 70)
    
    # Initialize analysis
    analysis = NetflixKaggleAnalysis()
    
    # Run pipeline
    result = (analysis
              .load_data())
    
    if result is None:
        print("\n⚠️  Cannot proceed without dataset.")
        print("Please download the dataset and run again.")
        return
    
    (analysis
     .preprocess_data()
     .explore_data()
     .find_optimal_clusters()
     .perform_clustering()
     .visualize_clusters()
     .analyze_clusters())
    
    print("\n🎉 Analysis Complete!")
    print(f"Total shows analyzed: {len(analysis.df_processed):,}")
    print(f"Clusters found: {analysis.optimal_clusters}")
    print(f"Silhouette Score: {silhouette_score(analysis.X_scaled, analysis.df_processed['cluster']):.3f}")

if __name__ == "__main__":
    main()
