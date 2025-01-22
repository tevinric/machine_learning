# Insurance Policy Clustering Tutorial Using DBSCAN
## A Step-by-Step Guide to Density-Based Clustering

### Introduction
This tutorial demonstrates how to use DBSCAN (Density-Based Spatial Clustering of Applications with Noise) for clustering insurance policies. DBSCAN has several advantages over K-means:
- Automatically determines the number of clusters
- Can find clusters of arbitrary shapes
- Identifies outliers as noise points
- Doesn't assume spherical clusters

### Prerequisites
```python
pip install pandas numpy scikit-learn seaborn matplotlib category_encoders kneed
```

### Step 1: Import Libraries and Setup
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import TargetEncoder
from sklearn.decomposition import PCA
from kneed import KneeLocator
import warnings
warnings.filterwarnings('ignore')
```

### Step 2: Data Generation
[Previous data generation code remains the same as it's independent of the clustering algorithm]

### Step 3: Feature Engineering with DBSCAN-Specific Considerations
```python
def engineer_features_for_dbscan(df):
    """
    Perform feature engineering with considerations for density-based clustering
    """
    # Basic feature engineering (same as before)
    df[['Vehicle_Year', 'Vehicle_Make', 'Vehicle_Model']] = df['Vehicle'].str.split(' ', n=2, expand=True)
    df['Vehicle_Year'] = df['Vehicle_Year'].astype(int)
    df['Vehicle_Age'] = 2024 - df['Vehicle_Year']
    
    # Geographic risk scoring
    suburb_risk_mapping = {
        'Central Business District': 0.8,
        'Northern Suburbs': 0.6,
        'Eastern Suburbs': 0.7,
        'Western Suburbs': 0.5,
        'Southern Suburbs': 0.55
    }
    
    df['Suburb_Risk'] = df['Suburb'].map(suburb_risk_mapping)
    
    # Enhanced risk score for density-based clustering
    df['Risk_Score'] = (
        df['Claims'] * 20 +
        (df['Forced_Cancelled'] == 'Yes') * 30 +
        (df['Convictions'] == 'Yes') * 40 +
        (df['Performance_Enhanced'] == 'Yes') * 15 -
        df['NCB'] * 5 +
        df['Suburb_Risk'] * 10
    ).clip(0, 100)
    
    # Create premium density score for better DBSCAN clustering
    df['Premium_Density'] = df.groupby('Suburb')['Premium'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Create risk density score
    df['Risk_Density'] = df.groupby('Suburb')['Risk_Score'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    return df

```

### Step 4: Enhanced Data Preprocessing for DBSCAN
```python
def preprocess_for_dbscan(df, target_cols=['Vehicle_Make', 'Vehicle_Model', 'Suburb', 'Postal_Code']):
    """
    Preprocess data specifically for DBSCAN clustering
    """
    df_processed = df.copy()
    
    # Target encode high-cardinality features
    target_encoder = TargetEncoder()
    df_processed[target_cols] = target_encoder.fit_transform(df_processed[target_cols], df_processed['Premium'])
    
    # Label encode categorical variables
    categorical_cols = ['Gender', 'Marital_Status', 'Parking_Condition', 'Regular_Driver',
                       'Vehicle_Use', 'Performance_Enhanced', 'Forced_Cancelled',
                       'Convictions', 'License_Type']
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    # Scale numerical features
    numerical_cols = ['Age', 'Credit_Score', 'NCB', 'Claims', 'Excess', 'Premium',
                     'Vehicle_Age', 'Risk_Score', 'Premium_Density', 'Risk_Density'] + target_cols
    
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    return df_processed, scaler, label_encoders, target_encoder
```

### Step 5: DBSCAN Parameter Selection
```python
def find_optimal_epsilon(X, min_samples):
    """
    Find optimal epsilon using the elbow method
    """
    neigh = NearestNeighbors(n_neighbors=min_samples)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances[:, min_samples-1])
    
    # Find the elbow point
    kneedle = KneeLocator(
        range(len(distances)), 
        distances,
        S=1.0,
        curve='convex',
        direction='increasing'
    )
    
    return distances[kneedle.elbow] if kneedle.elbow else np.mean(distances)

def tune_dbscan(X, min_samples_range=range(5, 51, 5)):
    """
    Tune DBSCAN parameters
    """
    results = []
    
    for min_samples in min_samples_range:
        # Find optimal epsilon for current min_samples
        epsilon = find_optimal_epsilon(X, min_samples)
        
        # Fit DBSCAN
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        clusters = dbscan.fit_predict(X)
        
        # Skip if all points are noise
        if len(np.unique(clusters[clusters != -1])) < 2:
            continue
        
        # Calculate metrics
        n_clusters = len(np.unique(clusters[clusters != -1]))
        noise_points = np.sum(clusters == -1)
        
        # Calculate silhouette score only for non-noise points
        non_noise_mask = clusters != -1
        if np.sum(non_noise_mask) > 1:
            silhouette = silhouette_score(
                X[non_noise_mask], 
                clusters[non_noise_mask]
            )
        else:
            silhouette = -1
        
        results.append({
            'min_samples': min_samples,
            'epsilon': epsilon,
            'n_clusters': n_clusters,
            'noise_points': noise_points,
            'noise_ratio': noise_points / len(X),
            'silhouette': silhouette
        })
    
    return pd.DataFrame(results)
```

### Step 6: DBSCAN Model Training and Analysis
```python
def train_dbscan(X, epsilon, min_samples):
    """
    Train DBSCAN model with optimal parameters
    """
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    clusters = dbscan.fit_predict(X)
    
    return dbscan, clusters

def analyze_dbscan_clusters(df, clusters):
    """
    Analyze DBSCAN clustering results
    """
    df_analysis = df.copy()
    df_analysis['Cluster'] = clusters
    
    # Separate noise points
    noise_mask = clusters == -1
    print(f"Number of noise points: {np.sum(noise_mask)}")
    print(f"Noise ratio: {np.sum(noise_mask) / len(clusters):.2%}")
    
    # Calculate cluster statistics
    cluster_stats = df_analysis[~noise_mask].groupby('Cluster').agg({
        'Premium': ['mean', 'std', 'min', 'max'],
        'Age': 'mean',
        'Credit_Score': 'mean',
        'NCB': 'mean',
        'Claims': 'mean',
        'Risk_Score': 'mean'
    }).round(2)
    
    # Analyze noise points
    noise_stats = df_analysis[noise_mask].agg({
        'Premium': ['mean', 'std', 'min', 'max'],
        'Age': 'mean',
        'Credit_Score': 'mean',
        'NCB': 'mean',
        'Claims': 'mean',
        'Risk_Score': 'mean'
    }).round(2)
    
    return cluster

# Save model and preprocessors
import joblib

def save_model(model, scaler, label_encoders, target_encoder, base_path='models/'):
    """
    Save all model components
    """
    joblib.dump(model, f'{base_path}dbscan_model.pkl')
    joblib.dump(scaler, f'{base_path}scaler.pkl')
    joblib.dump(label_encoders, f'{base_path}label_encoders.pkl')
    joblib.dump(target_encoder, f'{base_path}target_encoder.pkl')

def load_model(base_path='models/'):
    """
    Load all model components
    """
    model = joblib.load(f'{base_path}dbscan_model.pkl')
    scaler = joblib.load(f'{base_path}scaler.pkl')
    label_encoders = joblib.load(f'{base_path}label_encoders.pkl')
    target_encoder = joblib.load(f'{base_path}target_encoder.pkl')
    
    return model, scaler, label_encoders, target_encoder

### Step 11: Model Monitoring and Maintenance

```python
def monitor_cluster_distribution(new_predictions, historical_distribution):
    """
    Monitor changes in cluster distribution and noise ratio
    """
    new_dist = pd.Series(new_predictions).value_counts(normalize=True)
    
    # Calculate distribution drift
    common_clusters = set(new_dist.index) & set(historical_distribution.index)
    drift = {
        cluster: abs(new_dist.get(cluster, 0) - historical_distribution.get(cluster, 0))
        for cluster in common_clusters | set(new_dist.index) | set(historical_distribution.index)
    }
    
    # Calculate noise ratio change
    new_noise_ratio = (new_predictions == -1).mean()
    hist_noise_ratio = historical_distribution.get(-1, 0)
    
    return {
        'distribution_drift': drift,
        'noise_ratio_change': new_noise_ratio - hist_noise_ratio
    }

def analyze_noise_points(df, clusters, threshold=0.1):
    """
    Analyze characteristics of noise points to understand outliers
    """
    df_analysis = df.copy()
    df_analysis['is_noise'] = clusters == -1
    
    # Analyze feature distributions for noise vs. non-noise points
    analysis_results = {}
    
    for column in df_analysis.select_dtypes(include=[np.number]).columns:
        noise_stats = df_analysis[df_analysis['is_noise']][column].describe()
        non_noise_stats = df_analysis[~df_analysis['is_noise']][column].describe()
        
        # Calculate relative difference
        mean_diff = abs(noise_stats['mean'] - non_noise_stats['mean']) / non_noise_stats['mean']
        
        if mean_diff > threshold:
            analysis_results[column] = {
                'noise_mean': noise_stats['mean'],
                'non_noise_mean': non_noise_stats['mean'],
                'relative_difference': mean_diff
            }
    
    return analysis_results

def check_data_drift(new_data, reference_data, threshold=0.1):
    """
    Check for data drift in new policies
    """
    drift_detected = False
    drift_metrics = {}
    
    # Check numerical features
    for column in new_data.select_dtypes(include=[np.number]).columns:
        new_mean = new_data[column].mean()
        ref_mean = reference_data[column].mean()
        new_std = new_data[column].std()
        ref_std = reference_data[column].std()
        
        mean_drift = abs(new_mean - ref_mean) / ref_mean
        std_drift = abs(new_std - ref_std) / ref_std
        
        if mean_drift > threshold or std_drift > threshold:
            drift_detected = True
            drift_metrics[column] = {
                'mean_drift': mean_drift,
                'std_drift': std_drift
            }
    
    # Check categorical features
    for column in new_data.select_dtypes(include=['object']).columns:
        new_dist = new_data[column].value_counts(normalize=True)
        ref_dist = reference_data[column].value_counts(normalize=True)
        
        dist_drift = np.mean(abs(new_dist - ref_dist))
        
        if dist_drift > threshold:
            drift_detected = True
            drift_metrics[column] = {
                'distribution_drift': dist_drift
            }
    
    return drift_detected, drift_metrics
```

### Step 12: Advantages and Limitations of DBSCAN for Insurance Clustering

#### Advantages:
1. Noise Detection:
   - Identifies outlier policies that don't fit standard risk profiles
   - Helps detect potentially fraudulent or high-risk cases
   - Useful for underwriting exception handling

2. Flexible Cluster Shapes:
   - Can find natural risk groupings without forcing spherical clusters
   - Better handles complex relationships between features
   - More realistic for real-world insurance data

3. Automatic Cluster Detection:
   - No need to specify number of clusters upfront
   - Adapts to natural patterns in the data
   - Useful for discovering emerging risk segments

#### Limitations:
1. Parameter Sensitivity:
   - Results highly dependent on epsilon and min_samples
   - May need frequent parameter tuning as data evolves
   - Requires domain expertise for parameter interpretation

2. Density Variations:
   - May struggle with clusters of varying densities
   - Could miss smaller but valid risk segments
   - Might need preprocessing to normalize density

3. Scalability:
   - O(n²) complexity in worst case
   - May need optimization for very large policy databases
   - Consider using HDBSCAN for better scalability

### Step 13: Best Practices for Insurance Application

1. Parameter Selection:
   ```python
   # Use domain knowledge for min_samples
   min_samples = max(5, len(features) * 2)  # Rule of thumb
   
   # Scale epsilon based on premium ranges
   epsilon = premium_std * 0.1  # Start with 10% of premium standard deviation
   ```

2. Feature Importance:
   ```python
   def analyze_feature_importance(df, clusters):
       """
       Analyze which features contribute most to cluster formation
       """
       feature_importance = {}
       base_clusters = clusters.copy()
       
       for column in df.columns:
           shuffled_df = df.copy()
           shuffled_df[column] = np.random.permutation(shuffled_df[column])
           
           new_clusters = DBSCAN(
               eps=epsilon, 
               min_samples=min_samples
           ).fit_predict(shuffled_df)
           
           # Compare clustering similarity
           similarity = np.mean(base_clusters == new_clusters)
           feature_importance[column] = 1 - similarity
       
       return pd.Series(feature_importance).sort_values(ascending=False)
   ```

3. Cluster Interpretation:
   ```python
   def interpret_clusters_for_underwriting(df, clusters):
       """
       Generate underwriting guidelines based on clusters
       """
       for cluster in np.unique(clusters):
           if cluster == -1:
               continue
               
           cluster_data = df[clusters == cluster]
           
           # Generate rules based on percentiles
           rules = {}
           for column in df.select_dtypes(include=[np.number]).columns:
               q25, q75 = cluster_data[column].quantile([0.25, 0.75])
               rules[column] = f'{q25:.2f} - {q75:.2f}'
           
           yield cluster, rules
   ```

### Step 14: Ongoing Monitoring and Maintenance

1. Set up regular monitoring:
```python
def setup_monitoring(model_version, base_path='monitoring/'):
    """
    Initialize monitoring for a model version
    """
    monitoring_config = {
        'version': model_version,
        'start_date': pd.Timestamp.now(),
        'metrics': {
            'noise_ratio_threshold': 0.15,
            'drift_threshold': 0.1,
            'performance_metrics': []
        }
    }
    
    # Save monitoring configuration
    with open(f'{base_path}monitoring_config.json', 'w') as f:
        json.dump(monitoring_config, f)
    
    return monitoring_config
```

2. Implement retraining triggers:
```python
def check_retraining_needed(
    monitoring_metrics, 
    noise_threshold=0.15, 
    drift_threshold=0.1):
    """
    Check if model retraining is needed
    """
    retraining_needed = False
    reasons = []
    
    # Check noise ratio
    if monitoring_metrics['noise_ratio'] > noise_threshold:
        retraining_needed = True
        reasons.append(f'Noise ratio ({monitoring_metrics["noise_ratio"]:.2f}) above threshold')
    
    # Check data drift
    if monitoring_metrics['drift_score'] > drift_threshold:
        retraining_needed = True
        reasons.append(f'Data drift ({monitoring_metrics["drift_score"]:.2f}) above threshold')
    
    return retraining_needed, reasons
```

3. Performance evaluation:
```python
def evaluate_clustering_performance(df, clusters, business_metrics):
    """
    Evaluate clustering performance using business metrics
    """
    performance = {}
    
    # Calculate premium segmentation effectiveness
    premium_variance = df.groupby(clusters)['Premium'].var()
    performance['premium_segmentation'] = premium_variance.mean()
    
    # Calculate risk separation
    risk_separation = silhouette_score(
        df[['Premium', 'Risk_Score']], 
        clusters
    )
    performance['risk_separation'] = risk_separation
    
    # Add business metrics
    performance.update(business_metrics)
    
    return performance
```

This completes our comprehensive tutorial on using DBSCAN for insurance policy clustering. The key differences from K-means include:
- Better handling of outliers through noise point identification
- No need to specify number of clusters
- More flexible cluster shapes
- Different parameter tuning approach

Would you like me to explain any specific part in more detail?_stats, noise_stats
```

### Step 7: Enhanced Visualization for DBSCAN
```python
def plot_dbscan_analysis(df, clusters):
    """
    Create visualizations for DBSCAN clustering results
    """
    df_plot = df.copy()
    df_plot['Cluster'] = clusters
    
    # Premium distribution by cluster (including noise)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_plot, x='Cluster', y='Premium')
    plt.title('Premium Distribution by Cluster (Cluster -1 is Noise)')
    plt.show()
    
    # Risk Score vs Premium with noise points highlighted
    plt.figure(figsize=(10, 6))
    colors = np.where(clusters == -1, 'red', plt.cm.viridis(clusters / max(clusters)))
    plt.scatter(df_plot['Risk_Score'], df_plot['Premium'], 
               c=colors, alpha=0.6)
    plt.xlabel('Risk Score')
    plt.ylabel('Premium')
    plt.title('Risk Score vs Premium (Red points are Noise)')
    plt.show()
    
    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_plot.select_dtypes(np.number))
    
    plt.figure(figsize=(10, 6))
    colors = np.where(clusters == -1, 'red', plt.cm.viridis(clusters / max(clusters)))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6)
    plt.title('PCA Visualization of Clusters (Red points are Noise)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()
```

### Step 8: Model Validation and Stability Analysis
```python
def validate_dbscan(train_data, test_data, epsilon, min_samples):
    """
    Validate DBSCAN clustering on test set
    """
    # Train model
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    train_clusters = dbscan.fit_predict(train_data)
    
    # Predict test set
    test_clusters = dbscan.fit_predict(test_data)
    
    # Compare distributions
    train_noise_ratio = np.mean(train_clusters == -1)
    test_noise_ratio = np.mean(test_clusters == -1)
    
    train_n_clusters = len(np.unique(train_clusters[train_clusters != -1]))
    test_n_clusters = len(np.unique(test_clusters[test_clusters != -1]))
    
    return {
        'train_noise_ratio': train_noise_ratio,
        'test_noise_ratio': test_noise_ratio,
        'train_n_clusters': train_n_clusters,
        'test_n_clusters': test_n_clusters
    }

def check_dbscan_stability(X, epsilon, min_samples, n_iterations=10):
    """
    Check stability of DBSCAN clustering
    """
    results = []
    base_dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    base_clusters = base_dbscan.fit_predict(X)
    
    for i in range(n_iterations):
        # Add small random noise to data
        noise = np.random.normal(0, 0.1, X.shape)
        X_noisy = X + noise
        
        # Fit DBSCAN
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_noisy)
        
        # Compare with base clustering
        noise_ratio = np.mean(clusters == -1)
        n_clusters = len(np.unique(clusters[clusters != -1]))
        
        results.append({
            'iteration': i,
            'noise_ratio': noise_ratio,
            'n_clusters': n_clusters
        })
    
    return pd.DataFrame(results)
```

### Step 9: Putting It All Together
```python
# Generate and prepare data
df = generate_sample_data(n_samples=10000)
df = engineer_features_for_dbscan(df)

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Preprocess data
df_processed, scaler, label_encoders, target_encoder = preprocess_for_dbscan(train_df)

# Tune DBSCAN parameters
tuning_results = tune_dbscan(df_processed)
print("\nParameter Tuning Results:")
print(tuning_results.sort_values('silhouette', ascending=False).head())

# Select best parameters
best_params = tuning_results.loc[tuning_results['silhouette'].idxmax()]
epsilon = best_params['epsilon']
min_samples = int(best_params['min_samples'])

# Train final model
dbscan_model, clusters = train_dbscan(df_processed, epsilon, min_samples)

# Analyze results
cluster_stats, noise_stats = analyze_dbscan_clusters(train_df, clusters)
print("\nCluster Statistics:")
print(cluster_stats)
print("\nNoise Point Statistics:")
print(noise_stats)

# Visualize results
plot_dbscan_analysis(train_df, clusters)

# Validate model
validation_results = validate_dbscan(
    df_processed,
    preprocess_for_dbscan(test_df)[0],
    epsilon,
    min_samples
)
print("\nValidation Results:")
print(validation_results)

# Check stability
stability_results = check_dbscan_stability(df_processed, epsilon, min_samples)
print("\nStability Analysis:")
print(stability_results.describe())
```

### Step 10: Production Implementation
```python
def predict_cluster(new_data, model, scaler, label_encoders, target_encoder):
    """
    Predict cluster for new insurance policies
    """
    # Preprocess new data
    processed_data, _, _, _ = preprocess_for_dbscan(
        new_data,
        scaler=scaler,
        label_encoders=label_encoders,
        target_encoder=target_encoder
    )
    
    # Predict cluster
    cluster = model.fit_predict(processed_data)[0]
    
    return cluster
