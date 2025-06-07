import pandas as pd
import numpy as np
import joblib
import os

def load_models():
    """
    Load the trained models and feature names from disk.
    """
    models = {}
    for model_name in ['linear_regression', 'random_forest', 'xgboost']:
        model_path = os.path.join('models', f'{model_name}.joblib')
        if os.path.exists(model_path):
            # Load the entire pipeline (model + preprocessor)
            models[model_name] = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model file {model_path} not found")
    
    feature_names_path = os.path.join('models', 'feature_names.joblib')
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError("Feature names file not found")
    
    feature_names = joblib.load(feature_names_path)
    
    return models, feature_names

def get_student_input():
    """
    Get student characteristics for grade prediction.
    """
    print("\n🎓 Student Performance Prediction System")
    print("=======================================")
    print("\nPlease enter the student's information:")
    
    inputs = {
        # Demographics
        'age': {
            'description': 'Age (15-21)',
            'values': None,
            'explanation': 'Student age between 15 and 21'
        },
        'gender': {
            'description': 'Gender',
            'values': ['M', 'F'],
            'explanation': 'M for Male, F for Female'
        },
        'address': {
            'description': 'Address Type',
            'values': ['U', 'R'],
            'explanation': 'U for Urban, R for Rural'
        },
        'Medu': {
            'description': "Mother's Education",
            'values': ['None', 'Primary', 'Secondary', 'Higher'], # Changed to string values
            'explanation': 'None: 0, Primary: 1, Secondary: 2, Higher: 3'
        },
        'Fedu': {
            'description': "Father's Education",
            'values': ['None', 'Primary', 'Secondary', 'Higher'], # Changed to string values
            'explanation': 'None: 0, Primary: 1, Secondary: 2, Higher: 3'
        },
        
        # Lifestyle
        'internet': {
            'description': 'Internet Access',
            'values': ['yes', 'no'],
            'explanation': 'Does the student have internet access at home?'
        },
        'romantic': {
            'description': 'Romantic Relationship',
            'values': ['yes', 'no'],
            'explanation': 'Is the student in a romantic relationship?'
        },
        'freetime': {
            'description': 'Free Time',
            'values': ['Very low', 'Low', 'Medium', 'High'], # Changed to string values
            'explanation': '1: Very low, 2: Low, 3: Medium, 4: High'
        },
        'health': {
            'description': 'Health Status',
            'values': ['Very poor', 'Poor', 'Good', 'Very good'], # Changed to string values
            'explanation': '1: Very poor, 2: Poor, 3: Good, 4: Very good'
        },
        
        # Academic
        'studytime': {
            'description': 'Study Time',
            'values': ['<2h', '2-5h', '5-10h', '>10h'], # Changed to string values
            'explanation': '1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h'
        },
        'failures': {
            'description': 'Past Failures',
            'values': None,
            'explanation': 'Number of past class failures (0-3)'
        },
        'schoolsup': {
            'description': 'School Support',
            'values': ['yes', 'no'],
            'explanation': 'Extra educational support from school?'
        },
        'famsup': {
            'description': 'Family Support',
            'values': ['yes', 'no'],
            'explanation': 'Educational support from family?'
        },
        'paid': {
            'description': 'Extra Paid Classes',
            'values': ['yes', 'no'],
            'explanation': 'Extra paid classes outside school?'
        },
        'absences': {
            'description': 'Absences',
            'values': None,
            'explanation': 'Number of school absences (0-50)'
        }
    }
    
    student_data = {}
    
    for field, info in inputs.items():
        print(f"\n{info['description']}")
        print(f"Explanation: {info['explanation']}")
        
        while True:
            try:
                if info['values'] is None:  # Numeric field
                    value = input(f"Enter value: ")
                    if field in ['age', 'failures']:
                        value = int(value)
                        if field == 'age' and not (15 <= value <= 21):
                            print("Age must be between 15 and 21")
                            continue
                        elif field == 'failures' and not (0 <= value <= 3):
                            print("Failures must be between 0 and 3")
                            continue
                    else:  # absences
                        value = float(value)
                        if not (0 <= value <= 50):
                            print("Absences must be between 0 and 50")
                            continue
                    student_data[field] = value
                    break
                else:  # Categorical field
                    print(f"Possible values: {', '.join(info['values'])}")
                    user_input = input("Enter value: ").strip()
                    
                    # Create a lowercase mapping for validation
                    lower_case_values = {v.lower(): v for v in info['values']}
                    
                    if user_input.lower() in lower_case_values:
                        # Store the original, correctly-cased value
                        student_data[field] = lower_case_values[user_input.lower()]
                        break
                    else:
                        print("Invalid value. Please try again.")
            except ValueError:
                print("Please enter a valid number")
    
    return pd.DataFrame([student_data])

def make_prediction(student_data, models, feature_names):
    """
    Predict final grade using all available models.
    """
    # Ensure all feature columns exist and are in correct order
    # The pipeline will handle one-hot encoding and scaling internally
    student_data = student_data[feature_names] # Ensure consistent order
    
    predictions = {}
    for model_name, pipeline in models.items(): # Iterate over pipelines
        pred = pipeline.predict(student_data)[0] # Predict using the pipeline
        # Ensure prediction is between 0 and 20
        pred = max(0, min(20, pred))
        predictions[model_name] = round(float(pred), 0)  # Round to nearest integer for grade
    
    return predictions

def main():
    """
    Main function to run the student performance prediction system.
    """
    try:
        models, feature_names = load_models()
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return
    
    while True:
        student_data = get_student_input()
        
        try:
            # Pass feature_names to get_student_input so it can order the DataFrame correctly
            # Or, more robustly, order the DataFrame here before passing to make_prediction
            student_data = student_data[feature_names] # Ensure input columns match training order
            predictions = make_prediction(student_data, models, feature_names)
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            continue
        
        # Display results
        print("\n📊 Predicted Final Grade (0-20 scale):")
        print("=============================")
        for model_name, prediction in predictions.items():
            print(f"{model_name.replace('_', ' ').title()}: {prediction}")
        
        # Calculate average prediction
        avg_prediction = np.mean(list(predictions.values()))
        print(f"\n📈 Average Predicted Grade: {round(avg_prediction, 0)}")
        
        # Provide interpretation
        print("\n💡 Interpretation:")
        if avg_prediction >= 16:
            print("Excellent performance expected!")
        elif avg_prediction >= 14:
            print("Good performance expected.")
        elif avg_prediction >= 10:
            print("Average performance expected.")
        else:
            print("May need additional support.")
        
        while True:
            choice = input("\nWould you like to make another prediction? (yes/no): ").lower()
            if choice in ['yes', 'no']:
                break
            print("Please enter 'yes' or 'no'")
        
        if choice == 'no':
            break
    
    print("\nThank you for using the Student Performance Prediction System! 🎓")

if __name__ == "__main__":
    main()