# Insurance Policy Clustering Tutorial
## A Step-by-Step Guide to Machine Learning Clustering

### Introduction
This tutorial will guide you through the process of creating a clustering model for insurance policies. We'll use K-means clustering to group policies into premium ranges while considering various risk factors, vehicle details, and geographic information.

### Prerequisites
- Python 3.8+
- Basic understanding of Python programming
- Familiarity with pandas and numpy
- Understanding of basic machine learning concepts

### Required Libraries
```python
pip install pandas numpy scikit-learn seaborn matplotlib category_encoders
```

### Step 1: Project Setup and Data Generation
First, let's set up our project structure and create synthetic data that mimics real-world insurance policies.

```python
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import TargetEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
```

### Step 2: Data Generation
We'll create a synthetic dataset that mirrors real-world insurance data patterns.

```python
def generate_sample_data(n_samples=10000):
    """
    Generate synthetic insurance policy data including geographic features
    """
    np.random.seed(42)
    
    # Generate suburb and postal code data
    suburbs = [
        ('Northern Suburbs', ['2113', '2114', '2115']),
        ('Eastern Suburbs', ['2024', '2025', '2026']),
        ('Western Suburbs', ['2150', '2151', '2152']),
        ('Southern Suburbs', ['2205', '2206', '2207']),
        ('Central Business District', ['2000', '2001', '2002'])
    ]
    
    locations = []
    for _ in range(n_samples):
        area, postcodes = suburbs[np.random.randint(0, len(suburbs))]
        postcode = np.random.choice(postcodes)
        locations.append((area, postcode))
    
    suburbs_list, postcodes_list = zip(*locations)
    
    # Generate demographic data
    age = np.random.normal(35, 12, n_samples).clip(18, 75)
    gender = np.random.choice(['M', 'F'], n_samples)
    marital_status = np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples)
    
    # Generate vehicle data
    vehicle_data = [
        ('Toyota', ['Corolla', 'Camry', 'RAV4', 'Fortuner']),
        ('Honda', ['Civic', 'Accord', 'CR-V', 'HR-V']),
        ('Volkswagen', ['Golf', 'Polo', 'Tiguan', 'T-Cross']),
        ('BMW', ['320i', '520i', 'X3', 'X5']),
        ('Mercedes', ['C200', 'E250', 'GLC300', 'GLE450'])
    ]
    
    vehicles = []
    for _ in range(n_samples):
        make, models = vehicle_data[np.random.randint(0, len(vehicle_data))]
        year = np.random.randint(2010, 2024)
        model = np.random.choice(models)
        vehicles.append(f"{year} {make} {model}")
    
    # Generate correlated features
    credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
    credit_score += np.where(np.array(suburbs_list) == 'Central Business District', 50, 0)
    credit_score = credit_score.clip(300, 850)
    
    ncb = np.random.poisson(3, n_samples).clip(0, 15)
    
    # Parking conditions correlated with suburb
    parking_probabilities = {
        'Central Business District': [0.5, 0.3, 0.15, 0.05],
        'Northern Suburbs': [0.2, 0.4, 0.3, 0.1],
        'Eastern Suburbs': [0.3, 0.4, 0.2, 0.1],
        'Western Suburbs': [0.1, 0.3, 0.4, 0.2],
        'Southern Suburbs': [0.15, 0.35, 0.35, 0.15]
    }
    
    parking = []
    for suburb in suburbs_list:
        probs = parking_probabilities[suburb]
        parking.append(np.random.choice(
            ['Highly secure', 'Secure', 'Fairly Secure', 'Not secure'],
            p=probs
        ))
    
    regular_driver = np.random.choice(['Yes', 'No'], n_samples, p=[0.8, 0.2])
    vehicle_use = np.random.choice(
        ['Private', 'Business', 'Other'], 
        n_samples, 
        p=[0.7, 0.25, 0.05]
    )
    
    performance_enhanced = np.random.choice(['Yes', 'No'], n_samples, p=[0.1, 0.9])
    claims = np.random.poisson(0.3, n_samples).clip(0, 5)
    forced_cancelled = np.random.choice(['Yes', 'No'], n_samples, p=[0.05, 0.95])
    convictions = np.random.choice(['Yes', 'No'], n_samples, p=[0.03, 0.97])
    license_type = np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    # Generate excess and premium
    base_excess = np.random.normal(5000, 1000, n_samples)
    excess_multiplier = (
        (claims + 1) * 
        (np.where(forced_cancelled == 'Yes', 1.5, 1)) * 
        (np.where(convictions == 'Yes', 2, 1))
    )
    excess = (base_excess * excess_multiplier).clip(2000, 20000)
    
    base_premium = np.random.normal(10000, 2000, n_samples)
    premium_multiplier = (
        (1 + claims * 0.3) *
        (np.where(forced_cancelled == 'Yes', 1.8, 1)) *
        (np.where(convictions == 'Yes', 2.5, 1)) *
        (np.where(performance_enhanced == 'Yes', 1.4, 1)) *
        (np.where(vehicle_use == 'Business', 1.3, 1)) *
        (np.where(parking == 'Not secure', 1.2, 1)) *
        (np.exp(-ncb * 0.1))
    )
    premium = (base_premium * premium_multiplier).clip(5000, 50000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Marital_Status': marital_status,
        'Vehicle': vehicles,
        'Suburb': suburbs_list,
        'Postal_Code': postcodes_list,
        'Credit_Score': credit_score,
        'NCB': ncb,
        'Parking_Condition': parking,
        'Regular_Driver': regular_driver,
        'Vehicle_Use': vehicle_use,
        'Performance_Enhanced': performance_enhanced,
        'Claims': claims,
        'Forced_Cancelled': forced_cancelled,
        'Convictions': convictions,
        'License_Type': license_type,
        'Excess': excess,
        'Premium': premium
    })
    
    return df
```

### Step 3: Feature Engineering
Next, we'll create meaningful features from our raw data.

```python
def engineer_features(df):
    """
    Perform feature engineering including geographic features
    """
    # Split vehicle information
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
    
    # Create comprehensive risk score
    df['Risk_Score'] = (
        df['Claims'] * 20 +
        (df['Forced_Cancelled'] == 'Yes') * 30 +
        (df['Convictions'] == 'Yes') * 40 +
        (df['Performance_Enhanced'] == 'Yes') * 15 -
        df['NCB'] * 5 +
        df['Suburb_Risk'] * 10
    ).clip(0, 100)
    
    return df
```

### Step 4: Data Preprocessing
We'll prepare our data for clustering by encoding categorical variables and scaling numerical features.

```python
def preprocess_data(df, target_cols=['Vehicle_Make', 'Vehicle_Model', 'Suburb', 'Postal_Code']):
    """
    Preprocess the data for clustering
    """
    df_processed = df.copy()
    
    # Target encode high-cardinality features
    target_encoder = TargetEncoder()
    df_processed[target_cols] = target_encoder.fit_transform(df_processed[target_cols], df_processed['Premium'])
    
    # Label encode other categorical variables
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
                     'Vehicle_Age', 'Risk_Score'] + target_cols
    
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    return df_processed, scaler, label_encoders, target_encoder
```

### Step 5: Model Selection and Hyperparameter Tuning
Let's implement our hyperparameter tuning process.

```python
def tune_kmeans(X, k_range=range(2, 11)):
    """
    Perform hyperparameter tuning for KMeans clustering
    """
    results = []
    
    for k in k_range:
        init_methods = ['k-means++', 'random']
        for init in init_methods:
            for n_init in [10, 20, 30]:
                kmeans = KMeans(
                    n_clusters=k,
                    init=init,
                    n_init=n_init,
                    random_state=42
                )
                
                clusters = kmeans.fit_predict(X)
                
                silhouette = silhouette_score(X, clusters)
                calinski = calinski_harabasz_score(X, clusters)
                davies = davies_bouldin_score(X, clusters)
                inertia = kmeans.inertia_
                
                results.append({
                    'n_clusters': k,
                    'init': init,
                    'n_init': n_init,
                    'silhouette': silhouette,
                    'calinski': calinski,
                    'davies': davies,
                    'inertia': inertia
                })
    
    return pd.DataFrame(results)
```

### Step 6: Cluster Analysis and Visualization
Let's create functions to analyze and visualize our clustering results.

```python
def analyze_clusters(df, clusters):
    """
    Analyze the characteristics of each cluster
    """
    df_analysis = df.copy()
    df_analysis['Cluster'] = clusters
    
    # Calculate cluster statistics
    cluster_stats = df_analysis.groupby('Cluster').agg({
        'Premium': ['mean', 'std', 'min', 'max'],
        'Age': 'mean',
        'Credit_Score': 'mean',
        'NCB': 'mean',
        'Claims': 'mean',
        'Risk_Score': 'mean'
    }).round(2)
    
    # Calculate categorical distributions
    categorical_cols = ['Vehicle_Use', 'Performance_Enhanced', 'Parking_Condition']
    cluster_profiles = {}
    
    for col in categorical_cols:
        cluster_profiles[col] = df_analysis.groupby('Cluster')[col].value_counts(normalize=True)
    
    return cluster_stats, cluster_profiles

def plot_cluster_analysis(df, clusters):
    """
    Create visualizations for cluster analysis
    """
    df_plot = df.copy()
    df_plot['Cluster'] = clusters
    
    # Premium distribution by cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_plot, x='Cluster', y='Premium')
    plt.title('Premium Distribution by Cluster')
    plt.show()
    
    # Risk Score vs Premium by cluster
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_plot['Risk_Score'], df_plot['Premium'], 
                         c=df_plot['Cluster'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('Risk Score')
    plt.ylabel('Premium')
    plt.title('Risk Score vs Premium by Cluster')
    plt.show()
```

### Step 7: Putting It All Together
Here's how to run the complete analysis:

```python
# Generate sample data
df = generate_sample_data(n_samples=10000)
print("Dataset shape:", df.shape)

# Engineer features
df = engineer_features(df)
print("\nFeatures engineered successfully")

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print("\nTraining set shape:", train_df.shape)
print("Test set shape:", test_df.shape)

# Preprocess data
df_processed, scaler, label_encoders, target_encoder = preprocess_data(train_df)
print("\nPreprocessing completed")

# Tune hyperparameters
tuning_results = tune_kmeans(df_processed)
print("\nHyperparameter Tuning Results:")
print(tuning_results.sort_values('silhouette', ascending=False).head())

# Train final model with best parameters
best_params = tuning_results.loc[tuning_results['silhouette'].idxmax()]

# Train final model with best parameters
final_kmeans = KMeans(
    n_clusters=int(best_params['n_clusters']),
    init=best_params['init'],
    n_init=int(best_params['n_init']),
    random_state=42
)

# Fit final model
clusters = final_kmeans.fit_predict(df_processed)

# Analyze results
cluster_stats, cluster_profiles = analyze_clusters(train_df, clusters)

print("\nCluster Statistics:")
print(cluster_stats)

# Visualize results
plot_cluster_analysis(train_df, clusters)

### Step 8: Interpreting the Results

#### Understanding the Evaluation Metrics:

1. Silhouette Score (-1 to 1):
   - Measures how similar points are to their own cluster compared to other clusters
   - Higher values are better
   - Interpretation:
     * -1: Points are in wrong clusters
     * 0: Clusters are overlapping
     * 1: Clusters are well-separated

2. Calinski-Harabasz Score:
   - Ratio of between-cluster to within-cluster variance
   - Higher values indicate better clustering
   - Good for comparing different numbers of clusters

3. Davies-Bouldin Score:
   - Measures average similarity between clusters
   - Lower values indicate better clustering
   - Useful for evaluating cluster density and separation

#### Analyzing Cluster Characteristics:

```python
def interpret_clusters(cluster_stats, cluster_profiles):
    """
    Interpret the characteristics of each cluster
    """
    for cluster in cluster_stats.index:
        print(f"\nCluster {cluster} Analysis:")
        print("-" * 50)
        
        # Premium range
        print(f"Premium Range: ${cluster_stats.loc[cluster, ('Premium', 'min')]:,.2f} - "
              f"${cluster_stats.loc[cluster, ('Premium', 'max')]:,.2f}")
        print(f"Average Premium: ${cluster_stats.loc[cluster, ('Premium', 'mean')]:,.2f}")
        
        # Risk profile
        print(f"\nRisk Profile:")
        print(f"Average Risk Score: {cluster_stats.loc[cluster, ('Risk_Score', 'mean')]:,.2f}")
        print(f"Average Claims: {cluster_stats.loc[cluster, ('Claims', 'mean')]:,.2f}")
        
        # Customer profile
        print(f"\nCustomer Profile:")
        print(f"Average Age: {cluster_stats.loc[cluster, ('Age', 'mean')]:,.2f}")
        print(f"Average Credit Score: {cluster_stats.loc[cluster, ('Credit_Score', 'mean')]:,.2f}")
        
        # Vehicle usage
        print("\nVehicle Usage Distribution:")
        usage_dist = cluster_profiles['Vehicle_Use'][cluster].sort_values(ascending=False)
        for usage, pct in usage_dist.items():
            print(f"{usage}: {pct:.1%}")

# Interpret the clusters
interpret_clusters(cluster_stats, cluster_profiles)
```

### Step 9: Model Validation

To validate our clustering model, we should:

1. Check cluster stability:
```python
def check_cluster_stability(X, best_model, n_iterations=10):
    """
    Check cluster stability by running multiple times
    """
    labels_list = []
    for i in range(n_iterations):
        kmeans = KMeans(
            n_clusters=best_model.n_clusters,
            init=best_model.init,
            n_init=best_model.n_init,
            random_state=i
        )
        labels_list.append(kmeans.fit_predict(X))
    
    # Calculate consistency between runs
    base_labels = labels_list[0]
    consistency_scores = []
    
    for labels in labels_list[1:]:
        score = np.mean(base_labels == labels)
        consistency_scores.append(score)
    
    return np.mean(consistency_scores)

stability_score = check_cluster_stability(df_processed, final_kmeans)
print(f"\nCluster Stability Score: {stability_score:.3f}")
```

2. Cross-validate on test set:
```python
def validate_on_test_set(test_df, scaler, label_encoders, target_encoder, final_model):
    """
    Validate clustering model on test set
    """
    # Preprocess test data
    test_processed, _, _, _ = preprocess_data(test_df)
    
    # Predict clusters
    test_clusters = final_model.predict(test_processed)
    
    # Calculate metrics
    silhouette = silhouette_score(test_processed, test_clusters)
    calinski = calinski_harabasz_score(test_processed, test_clusters)
    davies = davies_bouldin_score(test_processed, test_clusters)
    
    return {
        'silhouette': silhouette,
        'calinski': calinski,
        'davies': davies,
        'clusters': test_clusters
    }

test_results = validate_on_test_set(test_df, scaler, label_encoders, target_encoder, final_kmeans)
print("\nTest Set Metrics:")
for metric, value in test_results.items():
    if metric != 'clusters':
        print(f"{metric}: {value:.3f}")
```

### Step 10: Using the Model in Production

To use this model in production:

1. Save the model and preprocessors:
```python
import joblib

# Save models and preprocessors
joblib.dump(final_kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')
```

2. Create a prediction function:
```python
def predict_cluster(new_data, model, scaler, label_encoders, target_encoder):
    """
    Predict cluster for new insurance policies
    """
    # Preprocess new data
    processed_data, _, _, _ = preprocess_data(
        new_data,
        scaler=scaler,
        label_encoders=label_encoders,
        target_encoder=target_encoder
    )
    
    # Predict cluster
    cluster = model.predict(processed_data)[0]
    
    return cluster
```

### Step 11: Model Maintenance and Monitoring

To maintain the model's performance:

1. Monitor cluster distribution:
```python
def monitor_cluster_distribution(new_predictions, historical_distribution):
    """
    Monitor changes in cluster distribution
    """
    new_dist = pd.Series(new_predictions).value_counts(normalize=True)
    drift = np.abs(new_dist - historical_distribution).mean()
    
    return drift
```

2. Track cluster characteristics:
```python
def track_cluster_characteristics(new_data, clusters, historical_stats):
    """
    Track changes in cluster characteristics
    """
    new_stats, _ = analyze_clusters(new_data, clusters)
    
    # Calculate differences
    diff = new_stats - historical_stats
    
    return diff
```

### Final Notes:

1. Model Limitations:
   - K-means assumes spherical clusters
   - Sensitive to outliers
   - Requires number of clusters to be specified
   - May not capture complex relationships

2. Potential Improvements:
   - Try other clustering algorithms (DBSCAN, Hierarchical Clustering)
   - Add more features or engineer different features
   - Implement more sophisticated geographic analysis
   - Use ensemble clustering methods

3. Next Steps:
   - Create decision trees for each cluster
   - Implement real-time prediction API
   - Set up monitoring dashboard
   - Develop retraining pipeline
