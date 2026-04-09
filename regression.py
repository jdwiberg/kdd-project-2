# CS 4445 - Project 2
# Segment: Regression Techniques
# Group: Shane G. , Jake W. , Victor C.
# April 4th 2026

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from preprocessing import regression_pp
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy import stats


# Regression ------------------------------------------------------------------------------
def regression():
    X_train, X_test, y_train, y_test = regression_pp()



    # Majority Class Classifier - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    '''
    Library Function: DummyClassifier
    Parameters: strategy='mean' (others: default)
    Purpose: Make "Baseline" prediction on target value (), based on average target value output (Ignores input features)
    '''
    print("1). Creating Baseline")
    reg = DummyRegressor(strategy='mean'               # Create 'empty' model (switched from 'DummyClassifier' because the value is cont.

                        )
    reg.fit(X_train, y_train)                           # Train Model (a very dumb model)
    predictions = reg.predict(X_test)                   # Dumb model makes predictions
    score1 = reg.score(X_test, y_test)                  # Results
    print("Score: " , score1)


    # Linear Regression - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    '''
    Library Function: LinearRegression
    Parameters: Default
    Purpose: Find line of best fit, and produce prediction equation (sum of attributes with weighted coefficients)
    '''
    print("\n2).Linear Regression")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    corr, _ = stats.pearsonr(y_test, predictions)
    print("Correlation Coefficient: ", corr)
    score2 = model.score(X_test, y_test)
    print("Score: " , score2)

    # Regression Trees - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    '''
    Library Function: DecisionTreeRegressor
    Parameters: Default
    Purpose: 
    '''
    print("\n3). Decision Tree")
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    corr, _ = stats.pearsonr(y_test, predictions)
    print("Correlation Coefficient: ", corr)
    score3 = model.score(X_test, y_test)
    print("Score: " , score3)

    # Random Forest - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    '''
    Library Function: RandomForestClassifier
    Parameters: 
    Purpose: 
    '''
    print("\n4). Random Forest")

    print("\nExperiment #1: n_estimators = 100")
    model = RandomForestRegressor(n_estimators = 100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    corr, _ = stats.pearsonr(y_test, predictions)
    print("Correlation Coefficient: ", corr)
    score4a = model.score(X_test, y_test)
    print("Score: " , score4a)

    print("\nExperiment #2: n_estimators = 50")
    model = RandomForestRegressor(n_estimators = 50)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    corr, _ = stats.pearsonr(y_test, predictions)
    print("Correlation Coefficient: ", corr)
    score4b = model.score(X_test, y_test)
    print("Score: ", score4b)

    print("\nExperiment #3: max_depth = none")
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    corr, _ = stats.pearsonr(y_test, predictions)
    print("Correlation Coefficient: ", corr)
    score4c = model.score(X_test, y_test)
    print("Score: ", score4c)

    print("\nExperiment #4: max_depth = none")
    model = RandomForestRegressor(max_depth = 10)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    corr, _ = stats.pearsonr(y_test, predictions)
    print("Correlation Coefficient: ", corr)
    score4d = model.score(X_test, y_test)
    print("Score: ", score4d)

    print("\nRegression Section Complete")
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
    main()