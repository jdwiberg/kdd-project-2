from preprocessing import regression_pp # returns attributes and target (X, y) for regression dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def regression():
    X, y = regression_pp()
    return
    

def evaluation(y_pred, y_true, *, verbose=True):
    """
    Evaluates regression performance using various metrics.
    Returns a tuple of (correlation coefficient, MAE, MSE, RMSE, R^2).
    Takes in predicted values (y_pred) and true values (y_true).
    """
    correlation_coef = np.corrcoef(y_true, y_pred)[0, 1]
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    if verbose:
        print(f"Correlation Coefficient: {correlation_coef:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R^2 Score: {r2:.4f}")

    return (correlation_coef, mae, mse, rmse, r2)


def main():
    regression()
    
if __name__ == "__main__":
    regression()