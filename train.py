import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def generate_sample_data(n_samples=1000):
    """
    Generate sample student performance data with the specified columns.
    """
    np.random.seed(42)
    
    # Generate numerical values for calculation of G3
    age = np.random.randint(15, 22, n_samples)
    gender = np.random.choice(['M', 'F'], n_samples)
    address = np.random.choice(['U', 'R'], n_samples)
    medu_numeric = np.random.randint(0, 4, n_samples)
    fedu_numeric = np.random.randint(0, 4, n_samples)
    internet = np.random.choice(['yes', 'no'], n_samples)
    romantic = np.random.choice(['yes', 'no'], n_samples)
    freetime_numeric = np.random.randint(1, 5, n_samples)
    health_numeric = np.random.randint(1, 5, n_samples)
    studytime_numeric = np.random.randint(1, 5, n_samples)
    failures = np.random.randint(0, 4, n_samples)
    schoolsup = np.random.choice(['yes', 'no'], n_samples)
    famsup = np.random.choice(['yes', 'no'], n_samples)
    paid = np.random.choice(['yes', 'no'], n_samples)
    absences = np.random.randint(0, 51, n_samples)

    # Generate target variable (G3) using numerical values
    g3 = (
        10 +  # Base grade
        0.5 * (age - 15) +  # Age: 15-21 (0-6 points added)
        1.0 * medu_numeric +  # Medu: 0-3 (0-3 points added)
        1.0 * fedu_numeric +  # Fedu: 0-3 (0-3 points added)
        0.75 * studytime_numeric +  # Studytime: 1-4 (0.75-3 points added)
        -3.0 * failures +  # Failures: 0-3 (0 to -9 points subtracted)
        -0.1 * absences +  # Absences: 0-50 (0 to -5 points subtracted)
        np.random.normal(0, 1.5, n_samples)  # Slightly reduced noise
    )
    # Ensure G3 is between 0 and 20
    g3 = np.clip(g3, 0, 20)

    # Map numerical values to string categories for the DataFrame
    medu_map = {0: 'None', 1: 'Primary', 2: 'Secondary', 3: 'Higher'}
    freetime_map = {1: 'Very low', 2: 'Low', 3: 'Medium', 4: 'High'}
    health_map = {1: 'Very poor', 2: 'Poor', 3: 'Good', 4: 'Very good'}
    studytime_map = {1: '<2h', 2: '2-5h', 3: '5-10h', 4: '>10h'}

    data = {
        'age': age,
        'gender': gender,
        'address': address,
        'Medu': np.vectorize(medu_map.get)(medu_numeric),
        'Fedu': np.vectorize(medu_map.get)(fedu_numeric),
        'internet': internet,
        'romantic': romantic,
        'freetime': np.vectorize(freetime_map.get)(freetime_numeric),
        'health': np.vectorize(health_map.get)(health_numeric),
        'studytime': np.vectorize(studytime_map.get)(studytime_numeric),
        'failures': failures,
        'schoolsup': schoolsup,
        'famsup': famsup,
        'paid': paid,
        'absences': absences,
        'G3': g3
    }
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Ensure column order consistency with predict.py's expected inputs
    final_columns_order = [
        'age', 'gender', 'address', 'Medu', 'Fedu',
        'internet', 'romantic', 'freetime', 'health',
        'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'absences', 'G3'
    ]
    df = df[final_columns_order]
    
    return df

def load_and_preprocess_data(df):
    """
    Preprocess the student performance dataset.
    """
    # Separate features and target
    X = df.drop('G3', axis=1)
    y = df['G3']
    
    # Define categorical and numerical columns
    categorical_cols = [
        {'name': 'gender', 'categories': ['M', 'F']},
        {'name': 'address', 'categories': ['U', 'R']},
        {'name': 'Medu', 'categories': ['None', 'Primary', 'Secondary', 'Higher']},
        {'name': 'Fedu', 'categories': ['None', 'Primary', 'Secondary', 'Higher']},
        {'name': 'internet', 'categories': ['yes', 'no']},
        {'name': 'romantic', 'categories': ['yes', 'no']},
        {'name': 'freetime', 'categories': ['Very low', 'Low', 'Medium', 'High']},
        {'name': 'health', 'categories': ['Very poor', 'Poor', 'Good', 'Very good']},
        {'name': 'studytime', 'categories': ['<2h', '2-5h', '5-10h', '>10h']},
        {'name': 'schoolsup', 'categories': ['yes', 'no']},
        {'name': 'famsup', 'categories': ['yes', 'no']},
        {'name': 'paid', 'categories': ['yes', 'no']}
    ]
    numerical_cols = ['age', 'failures', 'absences']
    
    # Create preprocessing pipeline
    transformers = [
        ('num', StandardScaler(), numerical_cols)
    ]
    
    for col_info in categorical_cols:
        transformers.append((
            col_info['name'], 
            OneHotEncoder(drop='first', sparse_output=False, categories=[col_info['categories']]),
            [col_info['name']]
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough' # Keep other columns if any, though not expected here
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_models(X_train, y_train, preprocessor):
    """
    Train multiple models and return them along with the preprocessor.
    """
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'xgboost': XGBRegressor(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        # Create pipeline with preprocessing and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate models using various metrics.
    """
    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        }
    
    return results

def save_models_and_preprocessor(models, preprocessor, feature_names):
    """
    Save trained models and preprocessor to disk.
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save each model
    for name, model in models.items():
        # Save the entire pipeline, which includes the preprocessor
        joblib.dump(model, f'models/{name}.joblib')
    
    # The preprocessor is already part of the saved pipelines, no need to save it separately.
    # However, we still need to save feature names if they are to be used for consistency checks.
    # joblib.dump(preprocessor, 'models/preprocessor.joblib') # This line is no longer needed if pipeline is saved
    
    # Save feature names
    joblib.dump(feature_names, 'models/feature_names.joblib')

def main():
    """
    Main function to train and evaluate models.
    """
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data()
    
    # Load and preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(df)
    
    # Train models
    print("Training models...")
    models = train_models(X_train, y_train, preprocessor)
    
    # Evaluate models
    print("Evaluating models...")
    results = evaluate_models(models, X_test, y_test)
    
    # Print results
    print("\nModel Performance Metrics:")
    print("=========================")
    for model_name, metrics in results.items():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"R²: {metrics['R²']:.2f}")
    
    # Save models and preprocessor
    feature_names = X_train.columns.tolist()
    print(f"\nFeature names saved: {feature_names}")
    save_models_and_preprocessor(models, preprocessor, feature_names)
    print("\nModels and preprocessor saved successfully!")

if __name__ == "__main__":
    main() 