import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load the synthetic dataset.
    """
    try:
        data = pd.read_csv(file_path, sep=';')
        expected_columns = [
            'age', 'gender', 'address', 'Medu', 'Fedu',  # Demographics
            'internet', 'romantic', 'freetime', 'health',  # Lifestyle
            'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'absences',  # Academic
            'G3'  # Target
        ]
        if not all(col in data.columns for col in expected_columns):
            raise ValueError(f"Dataset must contain columns: {expected_columns}")
        print("Available columns:", data.columns.tolist())
        return data
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

def preprocess_data(data, target_column='G3', test_size=0.2, random_state=42):
    """
    Preprocess the dataset: handle missing values, scale features, and split into train/test sets.
    """
    # Validate target column
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data. Available columns: {data.columns.tolist()}")
    
    # Handle missing values
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    for column in numeric_columns:
        data[column] = data[column].fillna(data[column].median())
    
    # Define features and target
    features = [
        'age', 'gender', 'address', 'Medu', 'Fedu',  # Demographics
        'internet', 'romantic', 'freetime', 'health',  # Lifestyle
        'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'absences'  # Academic
    ]
    X = data[features]
    y = data[target_column]
    feature_names = X.columns
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, feature_names, scaler