```python
    def generate_data(self, n_samples=10000):
        """
        Generate synthetic insurance policy data
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        pd.DataFrame : Generated data
        """
        np.random.seed(self.config.random_state)
        
        # Generate locations
        suburbs = np.random.choice(
            self.known_categories['suburbs'],
            n_samples,
            p=[0.2, 0.25, 0.25, 0.15, 0.15]
        )
        
        # Generate postal codes based on suburbs
        postal_codes = []
        for suburb in suburbs:
            base_code = {
                'Central Business District': '2000',
                'Northern Suburbs': '2113',
                'Eastern Suburbs': '2024',
                'Western Suburbs': '2150',
                'Southern Suburbs': '2205'
            }[suburb]
            offset = np.random.randint(0, 3)
            postal_codes.append(f"{int(base_code) + offset}")
        
        # Generate vehicle data
        vehicles = []
        for _ in range(n_samples):
            make = np.random.choice(self.known_categories['vehicle_makes'])
            model = np.random.choice(self.known_categories['vehicle_models'][make])
            year = np.random.randint(2010, 2024)
            vehicles.append(f"{year} {make} {model}")
        
        # Generate correlated features
        age = np.random.normal(35, 12, n_samples).clip(18, 75)
        credit_scores = np.random.normal(650, 100, n_samples).clip(300, 850)
        
        # Add suburb-based adjustments
        for i, suburb in enumerate(suburbs):
            if suburb == 'Central Business District':
                credit_scores[i] += np.random.normal(50, 10)
        credit_scores = credit_scores.clip(300, 850)
        
        # Generate NCB with age correlation
        ncb = np.random.poisson(
            (age - 18) / 10
        ).clip(0, 15)
        
        # Generate parking conditions with suburb correlation
        parking_probs = {
            'Central Business District': [0.5, 0.3, 0.15, 0.05],
            'Northern Suburbs': [0.2, 0.4, 0.3, 0.1],
            'Eastern Suburbs': [0.3, 0.4, 0.2, 0.1],
            'Western Suburbs': [0.1, 0.3, 0.4, 0.2],
            'Southern Suburbs': [0.15, 0.35, 0.35, 0.15]
        }
        
        parking_conditions = [
            np.random.choice(
                self.known_categories['parking_conditions'],
                p=parking_probs[suburb]
            )
            for suburb in suburbs
        ]
        
        # Generate risk-correlated features
        claims = np.random.poisson(0.3, n_samples).clip(0, 5)
        forced_cancelled = np.random.choice(
            ['Yes', 'No'], 
            n_samples, 
            p=[0.05, 0.95]
        )
        convictions = np.random.choice(
            ['Yes', 'No'], 
            n_samples, 
            p=[0.03, 0.97]
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'Age': age,
            'Gender': np.random.choice(['M', 'F'], n_samples),
            'Marital_Status': np.random.choice(
                ['Single', 'Married', 'Divorced', 'Widowed'], 
                n_samples
            ),
            'Vehicle': vehicles,
            'Suburb': suburbs,
            'Postal_Code': postal_codes,
            'Credit_Score': credit_scores,
            'NCB': ncb,
            'Parking_Condition': parking_conditions,
            'Regular_Driver': np.random.choice(
                ['Yes', 'No'], 
                n_samples, 
                p=[0.8, 0.2]
            ),
            'Vehicle_Use': np.random.choice(
                self.known_categories['vehicle_usage'],
                n_samples,
                p=[0.7, 0.25, 0.05]
            ),
            'Performance_Enhanced': np.random.choice(
                ['Yes', 'No'], 
                n_samples, 
                p=[0.1, 0.9]
            ),
            'Claims': claims,
            'Forced_Cancelled': forced_cancelled,
            'Convictions': convictions,
            'License_Type': np.random.choice(
                self.known_categories['license_types'],
                n_samples
            )
        })
        
        # Generate risk-based premium
        df['Premium'] = self._calculate_premium(df)
        
        return df
    
    def _calculate_premium(self, df):
        """Calculate premium based on risk factors"""
        base_premium = np.random.normal(10000, 2000, len(df))
        
        multipliers = (
            (1 + df['Claims'] * 0.3) *
            (df['Forced_Cancelled'] == 'Yes').astype(float) * 1.8 +
            (df['Convictions'] == 'Yes').astype(float) * 2.5 +
            (df['Performance_Enhanced'] == 'Yes').astype(float) * 1.4 +
            (df['Vehicle_Use'] == 'Business').astype(float) * 1.3 +
            (df['Parking_Condition'] == 'Not secure').astype(float) * 1.2 +
            np.exp(-df['NCB'] * 0.1)
        )
        
        # Add suburb-based adjustments
        suburb_factors = df['Suburb'].map(self.config.suburb_risk_mapping)
        multipliers *= (1 + suburb_factors)
        
        # Add age-based adjustments
        age_factor = np.where(
            df['Age'] < 25,
            1.3,
            np.where(df['Age'] > 65, 1.2, 1.0)
        )
        multipliers *= age_factor
        
        # Calculate final premium
        premium = (base_premium * multipliers).clip(5000, 50000)
        
        return premium

class FeatureEngineering:
    """Feature engineering for insurance policy clustering"""
    
    def __init__(self, config):
        self.config = config
        self.feature_ranges = {}
        
    def transform(self, df):
        """
        Apply feature engineering transformations
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
            
        Returns:
        --------
        pd.DataFrame : Transformed data
        """
        df_transformed = df.copy()
        
        # Extract vehicle information
        if 'Vehicle' in df_transformed.columns:
            df_transformed[['Vehicle_Year', 'Vehicle_Make', 'Vehicle_Model']] = (
                df_transformed['Vehicle'].str.split(' ', n=2, expand=True)
            )
            df_transformed['Vehicle_Year'] = pd.to_numeric(
                df_transformed['Vehicle_Year'],
                errors='coerce'
            ).fillna(df_transformed['Vehicle_Year'].mode()[0])
            
            # Calculate vehicle age
            df_transformed['Vehicle_Age'] = 2024 - df_transformed['Vehicle_Year']
        
        # Create suburb risk score
        df_transformed['Suburb_Risk'] = (
            df_transformed['Suburb']
            .map(self.config.suburb_risk_mapping)
            .fillna(np.mean(list(self.config.suburb_risk_mapping.values())))
        )
        
        # Create comprehensive risk score
        weights = self.config.risk_score_weights
        df_transformed['Risk_Score'] = (
            df_transformed['Claims'].fillna(0) * weights['claims'] +
            (df_transformed['Forced_Cancelled'] == 'Yes').astype(int) * weights['forced_cancelled'] +
            (df_transformed['Convictions'] == 'Yes').astype(int) * weights['convictions'] +
            (df_transformed['Performance_Enhanced'] == 'Yes').astype(int) * weights['performance_enhanced'] -
            df_transformed['NCB'].fillna(0) * weights['ncb'] +
            df_transformed['Suburb_Risk'] * weights['suburb_risk']
        ).clip(0, 100)
        
        # Create density features
        for col in ['Premium', 'Risk_Score', 'Credit_Score']:
            df_transformed[f'{col}_Density'] = (
                df_transformed.groupby('Suburb')[col]
                .transform(lambda x: (x - x.mean()) / x.std().clip(lower=1e-6))
                .fillna(0)
            )
        
        # Store feature ranges for validation
        numerical_cols = [
            'Age', 'Credit_Score', 'NCB', 'Claims', 'Premium',
            'Risk_Score', 'Vehicle_Age'
        ]
        self.feature_ranges = {
            col: {
                'min': df_transformed[col].min(),
                'max': df_transformed[col].max(),
                'mean': df_transformed[col].mean(),
                'std': df_transformed[col].std()
            }
            for col in numerical_cols if col in df_transformed.columns
        }
        
        return df_transformed

class RobustPreprocessor:
    """Robust preprocessing with handling of unseen categories"""
    
    def __init__(self, config):
        self.config = config
        self.encoders = {}
        self.feature_names = None
        
    def fit_transform(self, df):
        """
        Fit preprocessor and transform data
        """
        df_processed = df.copy()
        
        # Handle high-cardinality features
        high_cardinality_cols = [
            col for col in df_processed.select_dtypes(include=['object']).columns
            if df_processed[col].nunique() > self.config.high_cardinality_threshold
        ]
        
        if high_cardinality_cols:
            self.encoders['target'] = TargetEncoder(
                handle_unknown='value',
                handle_missing='value'
            )
            df_processed[high_cardinality_cols] = (
                self.encoders['target']
                .fit_transform(df_processed[high_cardinality_cols], 
                             df_processed['Premium'])
            )
        
        # Handle other categorical features
        categorical_cols = [
            col for col in df_processed.select_dtypes(include=['object']).columns
            if col not in high_cardinality_cols
        ]
        
        self.encoders['label'] = {}
        for col in categorical_cols:
            le = LabelEncoder()
            # Add 'UNKNOWN' category during fitting
            unique_vals = df_processed[col].unique().tolist() + ['UNKNOWN']
            le.fit(unique_vals)
            df_processed[col] = le.transform(df_processed[col])
            self.encoders['label'][col] = le
        
        # Scale numerical features
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        self.encoders['scaler'] = StandardScaler()
        df_processed[numerical_cols] = (
            self.encoders['scaler']
            .fit_transform(df_processed[numerical_cols]
                         .fillna(df_processed[numerical_cols].mean()))
        )
        
        self.feature_names = df_processed.columns
        return df_processed
    
    def transform(self, df):
        """
        Transform new data using fitted preprocessor
        """
        df_processed = df.copy()
        
        # Handle high-cardinality features
        if 'target' in self.encoders:
            high_cardinality_cols = self.encoders['target'].get_feature_names_out()
            try:
                df_processed[high_cardinality_cols] = (
                    self.encoders['target']
                    .transform(df_processed[high_cardinality_cols])
                )
            except:
                # Fallback for new categories
                for col in high_cardinality_cols:
                    col_mean = self.encoders['target'].transform(
                        df_processed[high_cardinality_cols]
                    )[col].mean()
                    mask = ~df_processed[col].isin(
                        self.encoders['target'].get_feature_names_out()
                    )
                    df_processed.loc[mask, col] = col_mean
        
        # Handle other categorical features
        for col, le in self.encoders['label'].items():
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].map(
                    lambda x: x if x in le.classes_ else 'UNKNOWN'
                )
                df_processed[col] = le.transform(df_processed[col])
        
        # Scale numerical features
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numerical_cols] = (
            self.encoders['scaler']
            .transform(df_processed[numerical_cols]
                     .fillna(df_processed[numerical_cols].mean()))
        )
        
        return df_processed[self.feature_names]
