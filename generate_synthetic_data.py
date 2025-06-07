import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=1000, random_seed=42):
    """
    Generate a synthetic dataset with comprehensive student features.
    """
    np.random.seed(random_seed)
    
    # Demographics
    data = {
        'age': np.random.randint(15, 22, n_samples),  # Age range 15-21
        'gender': np.random.choice(['M', 'F'], n_samples),
        'address': np.random.choice(['U', 'R'], n_samples),  # Urban/Rural
        'Medu': np.random.randint(0, 4, n_samples),  # Mother's education (0-3)
        'Fedu': np.random.randint(0, 4, n_samples),  # Father's education (0-3)
        
        # Lifestyle
        'internet': np.random.choice(['yes', 'no'], n_samples),
        'romantic': np.random.choice(['yes', 'no'], n_samples),
        'freetime': np.random.randint(1, 5, n_samples),  # 1-4 scale
        'health': np.random.randint(1, 5, n_samples),  # 1-4 scale
        
        # Academic
        'studytime': np.random.randint(1, 5, n_samples),  # 1-4 scale
        'failures': np.random.randint(0, 4, n_samples),  # 0-3
        'schoolsup': np.random.choice(['yes', 'no'], n_samples),  # Extra educational support
        'famsup': np.random.choice(['yes', 'no'], n_samples),  # Family educational support
        'paid': np.random.choice(['yes', 'no'], n_samples),  # Extra paid classes
        'absences': np.random.uniform(0, 50, n_samples).round(0),  # 0-50
    }
    
    # Convert categorical variables to numeric
    df = pd.DataFrame(data)
    df['gender'] = (df['gender'] == 'M').astype(int)
    df['address'] = (df['address'] == 'U').astype(int)
    df['internet'] = (df['internet'] == 'yes').astype(int)
    df['romantic'] = (df['romantic'] == 'yes').astype(int)
    df['schoolsup'] = (df['schoolsup'] == 'yes').astype(int)
    df['famsup'] = (df['famsup'] == 'yes').astype(int)
    df['paid'] = (df['paid'] == 'yes').astype(int)
    
    # Generate G3 (final grade) as a function of inputs with some noise
    base_grade = 10.0  # Base grade of 10 (passing grade)
    
    df['G3'] = (
        base_grade +
        1.5 * df['Medu'] +  # Mother's education (0-3) -> +0 to +4.5
        1.5 * df['Fedu'] +  # Father's education (0-3) -> +0 to +4.5
        1.0 * df['studytime'] +  # Study time (1-4) -> +1 to +4
        -2.0 * df['failures'] +  # Past failures (0-3) -> -0 to -6
        -0.1 * df['absences'] +  # Absences (0-50) -> -0 to -5
        1.0 * df['internet'] +  # Internet access (0-1) -> +0 to +1
        0.5 * df['famsup'] +  # Family support (0-1) -> +0 to +0.5
        0.5 * df['schoolsup'] +  # School support (0-1) -> +0 to +0.5
        0.5 * df['paid'] +  # Extra paid classes (0-1) -> +0 to +0.5
        -0.5 * df['romantic'] +  # Romantic relationship (0-1) -> -0 to -0.5
        0.5 * df['health'] +  # Health (1-4) -> +0.5 to +2
        0.5 * df['freetime'] +  # Free time (1-4) -> +0.5 to +2
        np.random.normal(0, 0.5, n_samples)  # Random noise
    ).clip(0, 20).round(0)  # Scale 0-20
    
    # Save to CSV
    df.to_csv('synthetic_data.csv', sep=';', index=False)
    print("Synthetic dataset generated and saved as 'synthetic_data.csv'")
    print("Columns:", df.columns.tolist())
    return df

if __name__ == "__main__":
    generate_synthetic_data()