import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NetflixShowClustering:
    def __init__(self, data_path):
        """
        Initialize the Netflix Show Clustering analysis
        
        Args:
            data_path (str): Path to the Netflix shows CSV file
        """
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.kmeans = None
        self.optimal_clusters = None
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        self.df = pd.read_csv(self.data_path)
        print("Dataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nData types:")
        print(self.df.dtypes)
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        print(f"\nFirst few rows:")
        print(self.df.head())
        return self
    
    def explore_data(self):
        """Explore the dataset with descriptive statistics and visualizations"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic statistics
        print("\nDescriptive Statistics:")
        print(self.df.describe())
        
        # Genre distribution
        print(f"\nGenre Distribution:")
        print(self.df['genre'].value_counts())
        
        # Type distribution
        print(f"\nType Distribution:")
        print(self.df['type'].value_counts())
        
        # Create subplots for exploration
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Netflix Shows - Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # Rating distribution
        axes[0, 0].hist(self.df['rating'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Rating Distribution')
        axes[0, 0].set_xlabel('Rating')
        axes[0, 0].set_ylabel('Count')
        
        # Duration distribution
        axes[0, 1].hist(self.df['duration'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Duration Distribution')
        axes[0, 1].set_xlabel('Duration (minutes)')
        axes[0, 1].set_ylabel('Count')
        
        # Genre distribution
        genre_counts = self.df['genre'].value_counts()
        axes[0, 2].bar(range(len(genre_counts)), genre_counts.values, color='coral')
        axes[0, 2].set_title('Genre Distribution')
        axes[0, 2].set_xlabel('Genre')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_xticks(range(len(genre_counts)))
        axes[0, 2].set_xticklabels(genre_counts.index, rotation=45, ha='right')
        
        # Rating vs Duration scatter plot
        colors = ['red' if t == 'Movie' else 'blue' for t in self.df['type']]
        axes[1, 0].scatter(self.df['duration'], self.df['rating'], c=colors, alpha=0.6)
        axes[1, 0].set_title('Rating vs Duration')
        axes[1, 0].set_xlabel('Duration (minutes)')
        axes[1, 0].set_ylabel('Rating')
        axes[1, 0].legend(['Movie', 'TV Show'], loc='upper right')
        
        # Year distribution
        axes[1, 1].hist(self.df['year'], bins=15, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].set_title('Release Year Distribution')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Count')
        
        # Box plot of ratings by genre
        self.df.boxplot(column='rating', by='genre', ax=axes[1, 2])
        axes[1, 2].set_title('Rating Distribution by Genre')
        axes[1, 2].set_xlabel('Genre')
        axes[1, 2].set_ylabel('Rating')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return self
    
    def preprocess_data(self):
        """Preprocess data for clustering"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        self.df_processed = self.df.copy()
        
        # Handle missing values (if any)
        if self.df_processed.isnull().sum().sum() > 0:
            print("Handling missing values...")
            self.df_processed = self.df_processed.fillna(self.df_processed.mean(numeric_only=True))
        
        # Encode categorical variables
        categorical_cols = ['genre', 'type', 'country']
        for col in categorical_cols:
            if col in self.df_processed.columns:
                le = LabelEncoder()
                self.df_processed[f'{col}_encoded'] = le.fit_transform(self.df_processed[col])
                self.label_encoders[col] = le
                print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Select features for clustering
        clustering_features = ['rating', 'duration', 'genre_encoded', 'type_encoded', 'year', 'seasons']
        
        # Ensure all features exist
        available_features = [f for f in clustering_features if f in self.df_processed.columns]
        print(f"\nUsing features for clustering: {available_features}")
        
        # Create feature matrix
        X = self.df_processed[available_features].copy()
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        self.X_scaled = pd.DataFrame(X_scaled, columns=available_features)
        
        print(f"\nFeature matrix shape: {self.X_scaled.shape}")
        print(f"Scaled features summary:")
        print(self.X_scaled.describe())
        
        return self
    
    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        print("\n" + "="*50)
        print("FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("="*50)
        
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            sil_score = silhouette_score(self.X_scaled, kmeans.labels_)
            silhouette_scores.append(sil_score)
            print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={sil_score:.3f}")
        
        # Find optimal k using silhouette score
        self.optimal_clusters = K_range[np.argmax(silhouette_scores)]
        print(f"\nOptimal number of clusters: {self.optimal_clusters}")
        
        # Plot elbow curve and silhouette scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elbow method plot
        ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=self.optimal_clusters, color='red', linestyle='--', 
                   label=f'Optimal k = {self.optimal_clusters}')
        ax1.legend()
        
        # Silhouette score plot
        ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score for Different k')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=self.optimal_clusters, color='red', linestyle='--', 
                   label=f'Optimal k = {self.optimal_clusters}')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        return self
    
    def perform_clustering(self, n_clusters=None):
        """Perform K-Means clustering"""
        if n_clusters is None:
            n_clusters = self.optimal_clusters if self.optimal_clusters else 5
        
        print(f"\n" + "="*50)
        print(f"PERFORMING K-MEANS CLUSTERING (k={n_clusters})")
        print("="*50)
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(self.X_scaled)
        
        # Add cluster labels to original dataframe
        self.df_processed['cluster'] = cluster_labels
        
        # Calculate clustering metrics
        inertia = self.kmeans.inertia_
        silhouette_avg = silhouette_score(self.X_scaled, cluster_labels)
        
        print(f"Clustering completed!")
        print(f"Inertia: {inertia:.2f}")
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Cluster distribution:")
        print(self.df_processed['cluster'].value_counts().sort_index())
        
        return self
    
    def visualize_clusters(self):
        """Create comprehensive visualizations of the clustering results"""
        print("\n" + "="*50)
        print("CLUSTER VISUALIZATION")
        print("="*50)
        
        # PCA for 2D visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        
        # Create a comprehensive visualization
        fig = plt.figure(figsize=(20, 15))
        
        # 1. PCA scatter plot
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=self.df_processed['cluster'], 
                            cmap='tab10', alpha=0.7, s=50)
        ax1.set_title('Clusters in PCA Space')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter, ax=ax1)
        
        # 2. Rating vs Duration colored by clusters
        ax2 = plt.subplot(2, 3, 2)
        scatter2 = ax2.scatter(self.df_processed['duration'], 
                              self.df_processed['rating'], 
                              c=self.df_processed['cluster'], 
                              cmap='tab10', alpha=0.7, s=50)
        ax2.set_title('Rating vs Duration (Clustered)')
        ax2.set_xlabel('Duration (minutes)')
        ax2.set_ylabel('Rating')
        plt.colorbar(scatter2, ax=ax2)
        
        # 3. Cluster size distribution
        ax3 = plt.subplot(2, 3, 3)
        cluster_counts = self.df_processed['cluster'].value_counts().sort_index()
        bars = ax3.bar(cluster_counts.index, cluster_counts.values, 
                      color=plt.cm.tab10(np.linspace(0, 1, len(cluster_counts))))
        ax3.set_title('Cluster Size Distribution')
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Number of Shows')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 4. Average rating by cluster
        ax4 = plt.subplot(2, 3, 4)
        avg_rating = self.df_processed.groupby('cluster')['rating'].mean()
        bars4 = ax4.bar(avg_rating.index, avg_rating.values,
                       color=plt.cm.tab10(np.linspace(0, 1, len(avg_rating))))
        ax4.set_title('Average Rating by Cluster')
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('Average Rating')
        ax4.set_ylim(0, 10)
        
        # 5. Genre distribution by cluster
        ax5 = plt.subplot(2, 3, 5)
        genre_cluster = pd.crosstab(self.df_processed['cluster'], 
                                   self.df_processed['genre'])
        genre_cluster.plot(kind='bar', stacked=True, ax=ax5, 
                          colormap='tab20')
        ax5.set_title('Genre Distribution by Cluster')
        ax5.set_xlabel('Cluster')
        ax5.set_ylabel('Count')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=0)
        
        # 6. Type distribution by cluster
        ax6 = plt.subplot(2, 3, 6)
        type_cluster = pd.crosstab(self.df_processed['cluster'], 
                                  self.df_processed['type'])
        type_cluster.plot(kind='bar', ax=ax6, colormap='Set2')
        ax6.set_title('Type Distribution by Cluster')
        ax6.set_xlabel('Cluster')
        ax6.set_ylabel('Count')
        ax6.legend()
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.show()
        
        return self
    
    def analyze_clusters(self):
        """Provide detailed analysis of each cluster"""
        print("\n" + "="*50)
        print("CLUSTER ANALYSIS")
        print("="*50)
        
        n_clusters = len(self.df_processed['cluster'].unique())
        
        for cluster_id in range(n_clusters):
            cluster_data = self.df_processed[self.df_processed['cluster'] == cluster_id]
            
            print(f"\n{'='*30}")
            print(f"CLUSTER {cluster_id} ANALYSIS")
            print(f"{'='*30}")
            print(f"Size: {len(cluster_data)} shows ({len(cluster_data)/len(self.df_processed)*100:.1f}%)")
            
            # Statistical summary
            print(f"\nStatistical Summary:")
            print(f"Average Rating: {cluster_data['rating'].mean():.2f} (±{cluster_data['rating'].std():.2f})")
            print(f"Average Duration: {cluster_data['duration'].mean():.1f} minutes (±{cluster_data['duration'].std():.1f})")
            print(f"Year Range: {cluster_data['year'].min()} - {cluster_data['year'].max()}")
            
            # Most common characteristics
            print(f"\nMost Common Characteristics:")
            print(f"Genre: {cluster_data['genre'].mode().iloc[0] if not cluster_data['genre'].mode().empty else 'N/A'}")
            print(f"Type: {cluster_data['type'].mode().iloc[0] if not cluster_data['type'].mode().empty else 'N/A'}")
            print(f"Country: {cluster_data['country'].mode().iloc[0] if not cluster_data['country'].mode().empty else 'N/A'}")
            
            # Genre distribution in this cluster
            print(f"\nGenre Distribution:")
            genre_dist = cluster_data['genre'].value_counts()
            for genre, count in genre_dist.head().items():
                percentage = (count / len(cluster_data)) * 100
                print(f"  {genre}: {count} ({percentage:.1f}%)")
            
            # Sample shows from this cluster
            print(f"\nSample Shows:")
            sample_shows = cluster_data[['title', 'genre', 'rating', 'duration', 'type']].head(5)
            for _, show in sample_shows.iterrows():
                print(f"  • {show['title']} ({show['genre']}, {show['type']}) - Rating: {show['rating']}, Duration: {show['duration']}min")
        
        return self
    
    def get_cluster_insights(self):
        """Generate insights about the clustering results"""
        print("\n" + "="*50)
        print("KEY INSIGHTS")
        print("="*50)
        
        insights = []
        
        # Overall clustering quality
        silhouette_avg = silhouette_score(self.X_scaled, self.df_processed['cluster'])
        if silhouette_avg > 0.5:
            insights.append(f"✓ Good clustering quality (Silhouette Score: {silhouette_avg:.3f})")
        elif silhouette_avg > 0.3:
            insights.append(f"~ Moderate clustering quality (Silhouette Score: {silhouette_avg:.3f})")
        else:
            insights.append(f"✗ Poor clustering quality (Silhouette Score: {silhouette_avg:.3f})")
        
        # Cluster size balance
        cluster_sizes = self.df_processed['cluster'].value_counts()
        max_size = cluster_sizes.max()
        min_size = cluster_sizes.min()
        size_ratio = max_size / min_size if min_size > 0 else float('inf')
        
        if size_ratio <= 3:
            insights.append("✓ Well-balanced cluster sizes")
        else:
            insights.append(f"~ Imbalanced cluster sizes (ratio: {size_ratio:.1f}:1)")
        
        # Genre separation
        for cluster_id in sorted(self.df_processed['cluster'].unique()):
            cluster_data = self.df_processed[self.df_processed['cluster'] == cluster_id]
            dominant_genre = cluster_data['genre'].mode().iloc[0]
            genre_purity = (cluster_data['genre'] == dominant_genre).mean()
            
            if genre_purity >= 0.7:
                insights.append(f"✓ Cluster {cluster_id}: Strong {dominant_genre} focus ({genre_purity:.1%} purity)")
            elif genre_purity >= 0.5:
                insights.append(f"~ Cluster {cluster_id}: Moderate {dominant_genre} focus ({genre_purity:.1%} purity)")
        
        # Type separation
        type_separation = pd.crosstab(self.df_processed['cluster'], self.df_processed['type'])
        for cluster_id in type_separation.index:
            movies = type_separation.loc[cluster_id, 'Movie'] if 'Movie' in type_separation.columns else 0
            tv_shows = type_separation.loc[cluster_id, 'TV Show'] if 'TV Show' in type_separation.columns else 0
            total = movies + tv_shows
            
            if total > 0:
                if movies / total >= 0.8:
                    insights.append(f"✓ Cluster {cluster_id}: Primarily Movies ({movies/total:.1%})")
                elif tv_shows / total >= 0.8:
                    insights.append(f"✓ Cluster {cluster_id}: Primarily TV Shows ({tv_shows/total:.1%})")
        
        # Print insights
        for insight in insights:
            print(insight)
        
        # Recommendations
        print(f"\n📊 RECOMMENDATIONS:")
        if silhouette_avg < 0.3:
            print("• Consider adjusting the number of clusters or feature selection")
        if size_ratio > 5:
            print("• Some clusters might be too specific - consider merging similar clusters")
        
        print("• Use these clusters for personalized recommendations")
        print("• Content creators can identify popular content patterns in each cluster")
        print("• Marketing teams can target specific audience segments based on clusters")
        
        return self

def main():
    """Main function to run the complete Netflix Show Clustering analysis"""
    print("🎬 NETFLIX SHOW CLUSTERING ANALYSIS")
    print("="*60)
    
    # Initialize the clustering analysis
    netflix_clustering = NetflixShowClustering('/Users/ceydaakin/netflix/netflix_shows.csv')
    
    # Run the complete pipeline
    (netflix_clustering
     .load_data()
     .explore_data()
     .preprocess_data()
     .find_optimal_clusters()
     .perform_clustering()
     .visualize_clusters()
     .analyze_clusters()
     .get_cluster_insights())
    
    print("\n🎉 Analysis completed successfully!")
    print("Check the generated visualizations and insights above.")

if __name__ == "__main__":
    main()