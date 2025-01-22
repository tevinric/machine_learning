# Detailed Explanation of Insurance Clustering Components

## 1. DBSCAN Algorithm Overview
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) works by:

### Core Concepts:
- **Epsilon (ε)**: The maximum distance between two points for them to be considered neighbors
- **MinPoints**: Minimum number of points required to form a dense region
- **Core Points**: Points that have at least MinPoints within ε distance
- **Border Points**: Points that are within ε distance of a core point but don't have enough neighbors
- **Noise Points**: Points that are neither core nor border points

### Algorithm Process:
1. For each point, find all points within ε distance
2. Identify core points that have at least MinPoints neighbors
3. Connect core points that are within ε distance of each other
4. Assign border points to their nearest core point's cluster
5. Label remaining points as noise

### Advantages for Insurance:
- Automatically identifies outlier policies (noise points)
- Finds natural risk groupings without forcing specific shapes
- Adapts to varying policy densities in feature space

## 2. Feature Engineering Details

### Basic Features:
1. **Age**: Raw policyholder age
   - Impact: Risk assessment and pricing
   - Processing: Standardized scaling

2. **Vehicle Information**:
   - Year: Extracted from full vehicle string
   - Make: Target encoded based on premium correlation
   - Model: Target encoded based on premium correlation
   - Vehicle Age: Calculated as (current year - vehicle year)

3. **Geographic Features**:
   - Suburb: Target encoded with premium correlation
   - Postal Code: Target encoded with premium correlation
   - Suburb Risk: Mapped risk scores based on historical data

### Engineered Features:
1. **Risk Score** (Composite feature):
```python
Risk_Score = (
    Claims * 20 +                              # Historical claims weight
    (Forced_Cancelled == 'Yes') * 30 +         # Cancellation penalty
    (Convictions == 'Yes') * 40 +              # Conviction penalty
    (Performance_Enhanced == 'Yes') * 15 -      # Vehicle modification risk
    NCB * 5 +                                  # No-claim bonus benefit
    Suburb_Risk * 10                           # Geographic risk factor
).clip(0, 100)
```

2. **Premium Density**:
```python
Premium_Density = (Premium - Suburb_Mean_Premium) / Suburb_Premium_Std
```
Purpose: Captures how a policy's premium compares to others in the same area

3. **Risk Density**:
```python
Risk_Density = (Risk_Score - Suburb_Mean_Risk) / Suburb_Risk_Std
```
Purpose: Identifies unusual risk profiles within geographic areas

## 3. Evaluation Metrics Explained

### 1. Silhouette Score (-1 to 1):
```python
silhouette = silhouette_score(X[non_noise_mask], clusters[non_noise_mask])
```
- Measures how similar points are to their own cluster vs other clusters
- Higher values indicate better-defined clusters
- Interpretation:
  * -1: Points are in wrong clusters
  * 0: Clusters are overlapping
  * 1: Clusters are well-separated

### 2. Noise Ratio:
```python
noise_ratio = np.sum(clusters == -1) / len(clusters)
```
- Proportion of policies identified as outliers
- Typical target range: 5-15%
- Higher values might indicate:
  * Too strict parameters (ε too small)
  * Data quality issues
  * Genuine outlier policies

### 3. Cluster Stability:
```python
def check_dbscan_stability(X, epsilon, min_samples, n_iterations=10):
    results = []
    for i in range(n_iterations):
        noise = np.random.normal(0, 0.1, X.shape)
        X_noisy = X + noise
        clusters = DBSCAN(eps=epsilon, min_samples=min_samples).fit_predict(X_noisy)
        results.append({
            'noise_ratio': np.mean(clusters == -1),
            'n_clusters': len(np.unique(clusters[clusters != -1]))
        })
    return pd.DataFrame(results)
```
- Measures clustering consistency under small data perturbations
- Important for reliable risk segmentation

## 4. Model Parameters and Tuning

### Parameter Selection Process:

1. **Epsilon (ε) Selection**:
```python
def find_optimal_epsilon(X, min_samples):
    # Calculate k-distance graph
    neigh = NearestNeighbors(n_neighbors=min_samples)
    nbrs = neigh.fit(X)
    distances = np.sort(nbrs.kneighbors(X)[0][:, min_samples-1])
    
    # Find elbow point
    kneedle = KneeLocator(
        range(len(distances)), 
        distances,
        curve='convex',
        direction='increasing'
    )
    return distances[kneedle.elbow]
```
- Uses k-distance graph to find natural density threshold
- Automatically adapts to data scale and density

2. **MinPoints Selection**:
- Rule of thumb: min_samples = 2 * n_features
- Adjusted based on dataset size and noise tolerance
- Typical range: 5-50 points

### Tuning Process:
1. Grid search over min_samples range
2. Calculate optimal epsilon for each min_samples
3. Evaluate clustering quality metrics
4. Select parameters balancing:
   - Noise ratio
   - Number of clusters
   - Cluster stability
   - Silhouette score

## 5. Business Metrics

### 1. Premium Segmentation:
```python
premium_variance = df.groupby(clusters)['Premium'].var()
segmentation_score = premium_variance.mean()
```
- Measures how well clusters separate different premium levels
- Lower variance indicates better premium segmentation

### 2. Risk Homogeneity:
```python
risk_separation = silhouette_score(
    df[['Premium', 'Risk_Score']], 
    clusters
)
```
- Evaluates how well clusters group similar risk profiles
- Higher scores indicate better risk separation

### 3. Geographic Concentration:
```python
geographic_concentration = df.groupby(['Cluster', 'Suburb']).size().unstack()
concentration_score = geographic_concentration.apply(lambda x: x.max() / x.sum())
```
- Measures geographic diversity within clusters
- Helps identify potential geographic bias

## 6. Monitoring and Maintenance

### Data Drift Detection:
```python
def check_data_drift(new_data, reference_data, threshold=0.1):
    drift_metrics = {}
    for column in new_data.columns:
        new_dist = new_data[column].value_counts(normalize=True)
        ref_dist = reference_data[column].value_counts(normalize=True)
        drift_metrics[column] = np.mean(abs(new_dist - ref_dist))
    return drift_metrics
```
- Monitors changes in feature distributions
- Triggers retraining when significant drift detected

### Performance Monitoring:
```python
def monitor_clustering_quality(new_predictions, historical_stats):
    metrics = {
        'noise_ratio': np.mean(new_predictions == -1),
        'n_clusters': len(np.unique(new_predictions[new_predictions != -1])),
        'cluster_sizes': pd.Series(new_predictions).value_counts()
    }
    return compare_with_historical(metrics, historical_stats)
```
- Tracks clustering quality over time
- Identifies degradation in performance

These components work together to create a robust clustering solution that:
1. Identifies natural risk segments
2. Handles outlier policies appropriately
3. Adapts to changes in the policy portfolio
4. Provides interpretable results for business use

Would you like me to explain any specific component in more detail?
