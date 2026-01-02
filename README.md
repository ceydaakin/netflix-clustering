# Netflix Show Clustering Analysis 🎬

A comprehensive machine learning project that groups similar Netflix shows using K-Means clustering based on genre, rating, duration, and other features.

## 📋 Project Overview

This project demonstrates unsupervised machine learning techniques to discover patterns in Netflix content. By clustering similar shows together, we can:

- Identify content patterns and preferences
- Enable personalized recommendations
- Support content acquisition decisions
- Optimize marketing strategies

## 🎯 Key Features

- **Data Exploration**: Comprehensive analysis of Netflix shows dataset
- **Feature Engineering**: Encoding categorical variables and scaling numerical features
- **Optimal Clustering**: Uses Elbow Method and Silhouette Analysis to find the best number of clusters
- **Rich Visualizations**: Multiple charts and plots to understand clustering results
- **Business Insights**: Actionable recommendations based on clustering patterns

## 🛠️ Tech Stack

- **Python 3.14+**
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **Seaborn & Matplotlib** - Data visualization
- **NumPy** - Numerical computing
- **Jupyter Notebook** - Interactive development

## 📊 Dataset Features

| Feature  | Type        | Description                                    |
| -------- | ----------- | ---------------------------------------------- |
| Title    | Text        | Name of the show/movie                         |
| Genre    | Categorical | Content category (Action, Drama, Comedy, etc.) |
| Rating   | Numerical   | IMDb rating (1-10 scale)                       |
| Duration | Numerical   | Length in minutes                              |
| Type     | Categorical | Movie or TV Show                               |
| Year     | Numerical   | Release year                                   |
| Seasons  | Numerical   | Number of seasons (for TV shows)               |
| Country  | Categorical | Country of origin                              |

## 🚀 Getting Started

### Prerequisites

1. Python 3.14+ installed
2. Virtual environment (recommended)

### Installation

1. **Clone the repository:**

   ```bash
   cd netflix
   ```

2. **Set up virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install pandas scikit-learn seaborn matplotlib numpy jupyter
   ```

### Running the Analysis

#### Option 1: Python Script

```bash
# Generate the dataset
python create_dataset.py

# Run the complete analysis
python netflix_clustering.py
```

#### Option 2: Jupyter Notebook (Recommended)

```bash
# Launch Jupyter
jupyter notebook

# Open netflix_clustering_analysis.ipynb
# Run all cells to see the interactive analysis
```

## 📈 Results Summary

The analysis typically identifies **2-3 main clusters** with distinct characteristics:

### Cluster Profiles Example:

- **Cluster 0**: Movies with moderate ratings (6.5-7.5), longer duration (90-150+ min)
- **Cluster 1**: TV Shows with high ratings (8.0+), shorter episodes (45-60 min)

### Key Metrics:

- **Silhouette Score**: 0.401 (Good clustering quality)
- **Optimal Clusters**: 2 (determined by silhouette analysis)
- **Content Distribution**: Well-separated Movies vs TV Shows

## 📊 Visualizations

The project includes comprehensive visualizations:

1. **Exploratory Data Analysis**

   - Rating distribution
   - Duration patterns
   - Genre popularity
   - Year trends

2. **Clustering Results**

   - PCA scatter plot
   - Cluster size distribution
   - Rating vs Duration (clustered)
   - Genre distribution by cluster

3. **Business Insights**
   - Cluster characteristics
   - Content type separation
   - Performance metrics

## 💡 Business Applications

### Content Strategy

- **Acquisition**: Focus on high-performing content types within each cluster
- **Production**: Develop content matching successful cluster patterns
- **Curation**: Create cluster-based content collections

### User Experience

- **Recommendations**: Suggest shows from same cluster as user preferences
- **Interface**: Design cluster-specific browsing experiences
- **Personalization**: Tailor marketing messages by cluster

### Analytics

- **Performance Tracking**: Monitor cluster evolution over time
- **Trend Analysis**: Identify emerging patterns in content preferences
- **Market Research**: Understand audience segmentation

## 📁 Project Structure

```
netflix/
├── create_dataset.py              # Dataset generation script
├── netflix_clustering.py          # Main analysis script
├── netflix_clustering_analysis.ipynb  # Interactive notebook
├── netflix_shows.csv             # Generated dataset
├── README.md                      # This file
└── .venv/                        # Virtual environment
```

## 🔍 Algorithm Details

### K-Means Clustering Process:

1. **Data Preprocessing**

   - Label encoding for categorical variables
   - Feature scaling using StandardScaler
   - Missing value handling

2. **Optimal Cluster Selection**

   - Elbow Method for inertia analysis
   - Silhouette Analysis for cluster quality
   - Cross-validation of results

3. **Feature Engineering**

   - Genre encoding (11 categories)
   - Content type binary encoding
   - Duration normalization
   - Year standardization

4. **Evaluation Metrics**
   - Within-cluster sum of squares (Inertia)
   - Silhouette coefficient
   - Cluster balance analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Netflix for inspiring the analysis concept
- Scikit-learn team for excellent machine learning tools
- Seaborn and Matplotlib for beautiful visualizations
- The open-source community for continuous innovation

## 📞 Contact

For questions or suggestions about this project:

- Create an issue on GitHub
- Email: [Your Email]
- LinkedIn: [Your LinkedIn]

---

**Made with ❤️ and lots of ☕**

_Last updated: December 2025_
