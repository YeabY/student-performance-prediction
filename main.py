from data_preprocessing import load_data, preprocess_data
from model import train_models, save_models_and_scaler
from evaluate import evaluate_models, plot_feature_importance, plot_predicted_vs_actual

def main():
    """
    Run the complete student performance prediction pipeline.
    """
    # Load and preprocess data
    data = load_data('synthetic_data.csv')
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(data, target_column='G3')
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Save models and scaler
    save_models_and_scaler(models, scaler, feature_names)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Print results
    for name, metrics in results.items():
        print(f"{name}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.2f}")
    
    # Generate visualizations
    plot_feature_importance(models['Random Forest'], feature_names)
    plot_predicted_vs_actual(y_test, models['Random Forest'].predict(X_test))

if __name__ == "__main__":
    main()