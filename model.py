from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib
import os
import numpy as np

def initialize_models():
    """
    Initialize machine learning models with default parameters.
    """
    return {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42, min_samples_leaf=5),
        'XGBoost': xgb.XGBRegressor(random_state=42, min_child_weight=5)
    }

def tune_random_forest(X_train, y_train):
    """
    Perform hyperparameter tuning for Random Forest using GridSearchCV.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15],
        'min_samples_leaf': [5, 10]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_models(X_train, y_train):
    """
    Train all models, including hyperparameter tuning for Random Forest.
    """
    models = initialize_models()
    models['Random Forest'] = tune_random_forest(X_train, y_train)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

def predict_with_constraints(model, X):
    """
    Make predictions and ensure they are within the valid range (0-20).
    """
    predictions = model.predict(X)
    return np.clip(predictions, 0, 20)

def save_models_and_scaler(models, scaler, feature_names):
    """
    Save trained models and scaler to disk.
    """
    os.makedirs('models', exist_ok=True)
    
    for name, model in models.items():
        model_path = os.path.join('models', f'{name.lower().replace(" ", "_")}.joblib')
        joblib.dump(model, model_path)
    
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(feature_names, 'models/feature_names.joblib')