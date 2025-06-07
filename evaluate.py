import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_models(models, X_test, y_test):
    """
    Evaluate models using MAE, RMSE, and R² metrics.
    """
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
    return results

def plot_feature_importance(model, feature_names, output_path='feature_importance.png'):
    """
    Plot feature importance for Random Forest model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[-5:]  # Only 5 features
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_predicted_vs_actual(y_test, y_pred, output_path='pred_vs_actual.png'):
    """
    Plot predicted vs actual grades.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 20], [0, 20], 'r--')  # 0-20 scale
    plt.xlabel('Actual Grade (0-20)')
    plt.ylabel('Predicted Grade (0-20)')
    plt.title('Predicted vs Actual Grade')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()