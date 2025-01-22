# Full Implementation and Usage

```python
class InsuranceClusteringPipeline:
    """Complete pipeline for insurance policy clustering"""
    
    def __init__(self, config):
        self.config = config
        self.data_generator = InsuranceDataGenerator(config)
        self.feature_engineer = FeatureEngineering(config)
        self.preprocessor = RobustPreprocessor(config)
        self.model = OptimizedKMeans(config)
        self.is_fitted = False
        
    def fit(self, df=None, n_samples=10000):
        """
        Fit the complete pipeline
        """
        if df is None:
            df = self.data_generator.generate_data(n_samples)
        
        # Split into train/test
        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        # Feature engineering
        train_engineered = self.feature_engineer.transform(train_df)
        
        # Preprocessing
        train_processed = self.preprocessor.fit_transform(train_engineered)
        
        # Optimize and fit model
        best_model, cv_scores = self.model.optimize(train_processed)
        
        # Process test data
        test_engineered = self.feature_engineer.transform(test_df)
        test_processed = self.preprocessor.transform(test_engineered)
        
        # Evaluate on test set
        test_clusters = best_model.predict(test_processed)
        test_silhouette = silhouette_score(test_processed, test_clusters)
        
        print("\nCross-validation Results:")
        print(f"Mean CV Silhouette Score: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
        print(f"Test Set Silhouette Score: {test_silhouette:.3f}")
        
        self.is_fitted = True
        return train_clusters, test_clusters
    
    def predict(self, df):
        """
        Predict clusters for new data
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted yet!")
        
        # Apply feature engineering
        df_engineered = self.feature_engineer.transform(df)
        
        # Apply preprocessing
        df_processed = self.preprocessor.transform(df_engineered)
        
        # Predict clusters
        clusters = self.model.best_model.predict(df_processed)
        
        return clusters, df_engineered
    
    def analyze_results(self, df, clusters):
        """
        Create comprehensive analysis of clustering results
        """
        df_analysis = df.copy()
        df_analysis['Cluster'] = clusters
        
        # Basic cluster statistics
        cluster_stats = df_analysis.groupby('Cluster').agg({
            'Premium': ['mean', 'std', 'min', 'max'],
            'Risk_Score': ['mean', 'std'],
            'Age': 'mean',
            'Credit_Score': 'mean',
            'NCB': 'mean',
            'Claims': 'mean'
        }).round(2)
        
        # Categorical distributions
        categorical_cols = [
            'Vehicle_Use', 'Parking_Condition', 'Suburb',
            'License_Type'
        ]
        
        cluster_profiles = {}
        for col in categorical_cols:
            if col in df_analysis.columns:
                cluster_profiles[col] = (
                    df_analysis
                    .groupby('Cluster')[col]
                    .value_counts(normalize=True)
                )
        
        return cluster_stats, cluster_profiles
    
    def plot_results(self, df, clusters):
        """
        Create comprehensive visualizations
        """
        df_plot = df.copy()
        df_plot['Cluster'] = clusters
        
        # 1. Premium distribution by cluster
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_plot, x='Cluster', y='Premium')
        plt.title('Premium Distribution by Cluster')
        plt.show()
        
        # 2. Risk Score vs Premium
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            df_plot['Risk_Score'],
            df_plot['Premium'],
            c=df_plot['Cluster'],
            cmap='viridis',
            alpha=0.6
        )
        plt.colorbar(scatter)
        plt.xlabel('Risk Score')
        plt.ylabel('Premium')
        plt.title('Risk Score vs Premium by Cluster')
        plt.show()
        
        # 3. PCA visualization
        pca = PCA(n_components=2)
        features = df_plot.select_dtypes(include=[np.number]).columns
        X_pca = pca.fit_transform(df_plot[features])
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=clusters,
            cmap='viridis',
            alpha=0.6
        )
        plt.colorbar(scatter)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA Visualization of Clusters')
        plt.show()

# Usage Example
if __name__ == "__main__":
    # Initialize configuration
    config = ClusteringConfig()
    
    # Create pipeline
    pipeline = InsuranceClusteringPipeline(config)
    
    # Generate and fit
    train_clusters, test_clusters = pipeline.fit(n_samples=10000)
    
    # Analyze results
    cluster_stats, cluster_profiles = pipeline.analyze_results(
        pipeline.data_generator.generate_data(1000),
        pipeline.predict(pipeline.data_generator.generate_data(1000))[0]
    )
    
    print("\nCluster Statistics:")
    print(cluster_stats)
    
    print("\nCluster Profiles:")
    for feature, distribution in cluster_profiles.items():
        print(f"\n{feature} Distribution:")
        print(distribution)
```

### Key Features of the Implementation:

1. Robust Data Generation:
   - Realistic correlations between features
   - Geographic patterns
   - Risk-based premium calculation

2. Advanced Feature Engineering:
   - Comprehensive risk scoring
   - Density features
   - Geographic risk factors

3. Robust Preprocessing:
   - Handling of unseen categories
   - Proper scaling
   - Missing value handling

4. Sophisticated Model Tuning:
   - Cross-validation
   - Hyperparameter optimization with Opt
