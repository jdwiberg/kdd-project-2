import argparse
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from scipy import stats
import time
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.utils import resample
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression


# Set the path to the file you'd like to load
file_path = "healthcare-dataset-stroke-data.csv"

# Load the latest version
dataframe = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "fedesoriano/stroke-prediction-dataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

FIGURES_DIR = Path("figures")

def model_filename(model):
    return str(model).replace("(", "_").replace(")", "").replace(", ", "_").replace("=", "-").replace(" ", "_")

def regression_pp(*, include_stroke = True): # use BMI for regression
    """
    Preprocesses the dataset for regression tasks. Returns features and target variable.
    If include_stroke is False, the 'stroke' feature will be dropped to prevent data leakage when predicting BMI.
    """
    # drop rows with missing BMI values, since we can't use them for regression
    X = dataframe.copy().dropna(subset=['bmi'])

    # remove the single "Other" gender row to keep gender binary
    X = X[X['gender'] != 'Other'].copy()

    # remove identifiers and the target before feature preprocessing
    X = X.drop(columns=['id'])

    # stroke may leak signal for BMI prediction, so exclude it
    if not include_stroke and 'stroke' in X.columns:
        X = X.drop(columns=['stroke'])

    # make 'Residence_type' into 'residence_type' to be consistent with other features
    X = X.rename(columns={'Residence_type': 'residence_type'})

    # encode binary categorical features as 0/1
    X['gender'] = X['gender'].map({'Female': 0, 'Male': 1})
    X['ever_married'] = X['ever_married'].map({'No': 0, 'Yes': 1})
    X['residence_type'] = X['residence_type'].map({'Rural': 0, 'Urban': 1})

    # one-hot encode the remaining categorical features
    X = X.join(
        pd.get_dummies(
            X[['work_type', 'smoking_status']],
            prefix=['work_type', 'smoking_status'],
            dtype=int,
        )
    ).drop(columns=['work_type', 'smoking_status'])

    y = X.pop('bmi')
    return X, y

def classification_pp(*, upsample = False): # use stroke for classification
    """
    Preprocesses the dataset for classification tasks. Returns features and target variable.
    If upsample is True, the minority class will be upsampled to address class imbalance.
    """
    X = dataframe.copy()

    # remove the single "Other" gender row to keep gender binary
    X = X[X['gender'] != 'Other'].copy()

    # remove identifiers and standardize the residence feature name
    X = X.drop(columns=['id']).rename(columns={'Residence_type': 'residence_type'})

    # fill missing BMI values with the median BMI
    X['bmi'] = X['bmi'].fillna(X['bmi'].median())

    # encode binary categorical features as 0/1
    X['gender'] = X['gender'].map({'Female': 0, 'Male': 1})
    X['ever_married'] = X['ever_married'].map({'No': 0, 'Yes': 1})
    X['residence_type'] = X['residence_type'].map({'Rural': 0, 'Urban': 1})

    # one-hot encode the remaining categorical features
    X = X.join(
        pd.get_dummies(
            X[['work_type', 'smoking_status']],
            prefix=['work_type', 'smoking_status'],
            dtype=int,
        )
    ).drop(columns=['work_type', 'smoking_status'])

    y = X.pop('stroke')

    if upsample:
        X_maj = X[y == 0]
        y_maj = y[y == 0]
        X_min = X[y == 1]
        y_min = y[y == 1]

        X_min_upsampled, y_min_upsampled = resample(
            X_min,
            y_min,
            replace=True,  # Sample with replacement
            n_samples=len(y_maj),  # Match number of maj samples
            random_state=42,
        )

        X = pd.concat([X_maj, X_min_upsampled])
        y = pd.concat([y_maj, y_min_upsampled])

    return X, y

def classification(*, save_graphs=False):
    """
    needs to find class probabilites, not predictions, so we can evaluate AUC and choose a threshold for precision/recall
    Use evaluation function to evaluate performance of different models
    """
    X, y = classification_pp()

    # Majority Class ==================================================================
    print("\n\n\n1). Majority Class Classifier\n--------------")
    model = DummyClassifier(strategy='most_frequent')
    start = time.perf_counter_ns()
    y_probs = cross_val_predict(model, X, y, cv=10, method='predict_proba')[:, 1]
    dt = time.perf_counter_ns() - start
    print(f"10-fold CV time: {dt / 1_000_000:.3f} ms")
    class_evaluation(y_probs, y, model, save_graphs=save_graphs)

    # Decision Tree ==================================================================
    print("\n2). Decision Tree\n------------------------")
    dt_models = [
        DecisionTreeClassifier(),
        DecisionTreeClassifier(max_depth=5),
        DecisionTreeClassifier(criterion='entropy'),
        DecisionTreeClassifier(splitter='random'),
        DecisionTreeClassifier(min_samples_split=10),
        
    ]

    for model in dt_models:
        print(f"\nModel: {model}")
        start = time.perf_counter_ns()
        y_probs = cross_val_predict(model, X, y, cv=10, method='predict_proba')[:, 1]
        dt = time.perf_counter_ns() - start
        print(f"10-fold CV time: {dt / 1_000_000:.3f} ms")
        model.fit(X, y)
        print("Node count:", model.tree_.node_count)
        print("Depth:", model.tree_.max_depth)
        print("Leaf count:", model.tree_.n_leaves)
        class_evaluation(y_probs, y, model, save_graphs=save_graphs)
        

    # Random Forests ==================================================================
    print("\nRandom Forest\n----------------------------")

    rf_models = [
        RandomForestClassifier(),
        RandomForestClassifier(criterion='entropy'),
        RandomForestClassifier(n_estimators=200),
        RandomForestClassifier(n_estimators=50),
        RandomForestClassifier(max_depth=10),
    ]

    for model in rf_models:
        print(f"\nModel: {model}")
        start = time.perf_counter_ns()
        y_probs = cross_val_predict(model, X, y, cv=10, method='predict_proba')[:, 1]
        dt = time.perf_counter_ns() - start
        print(f"10-fold CV time: {dt / 1_000_000:.3f} ms")
        class_evaluation(y_probs, y, model, save_graphs=save_graphs)

def class_evaluation(y_probs, y_true, model, *, threshold=0.5, verbose=True, graph=True, save_graphs=False):
    """
    Evaluates classification performance using various metrics.
    Returns a tuple of (accuracy, precision, recall, AUC, confusion matrix).
    Takes in predicted class PROBABILITIES (y_probs) and true labels (y_true), along with a threshold for converting probabilities to binary predictions.
    """
    y_pred = (y_probs >= threshold).astype(int)  # Convert probabilities to binary predictions

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0) # I've added the zero_division thingy to prevent warning doo-dad
    recall = recall_score(y_true, y_pred, zero_division=0) # Here too
    auc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred)

    if verbose:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC: {auc:.4f}")

    if graph:
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        plt.title(f"Confusion Matrix, {model}")
        if save_graphs:
            FIGURES_DIR.mkdir(exist_ok=True)
            plt.savefig(FIGURES_DIR / f"classification_confusion_matrix_{model_filename(model)}.png", bbox_inches="tight")
        plt.show()

    return (accuracy, precision, recall, auc, cm)


# Regression ------------------------------------------------------------------------------
def regression(*, save_graphs=False):
    X, y = regression_pp()

    # Majority Class Classifier - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    '''
    Library Function: DummyClassifier
    Parameters: strategy='mean' (others: default)
    Purpose: Make "Baseline" prediction on target value (), based on average target value output (Ignores input features)
    '''
    print("Majority Class Classifier\n---------------------")
    reg = DummyRegressor(strategy='mean')               # Create 'empty' model (switched from 'DummyClassifier' because the value is cont.
    start = time.perf_counter_ns()
    predictions = cross_val_predict(reg, X, y, cv=10)
    dt = time.perf_counter_ns() - start
    print(f"10-fold CV time: {dt / 1_000_000:.3f} ms")
    reg_evaluation(predictions, y, reg, save_graphs=save_graphs)


    # Linear Regression - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    '''
    Library Function: LinearRegression
    Parameters: Default
    Purpose: Find line of best fit, and produce prediction equation (sum of attributes with weighted coefficients)
    '''
    print("Linear Regression\n----------------")
    mdls = [LinearRegression(), LinearRegression(fit_intercept=False)]
    for model in mdls:
        print(f"Model: {model}")
        start = time.perf_counter_ns()
        predictions = cross_val_predict(model, X, y, cv=10)
        dt = time.perf_counter_ns() - start
        print(f"10-fold CV time: {dt / 1_000_000:.3f} ms")
        reg_evaluation(predictions, y, model, save_graphs=save_graphs)

    # Regression Trees - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    '''
    Library Function: DecisionTreeRegressor
    Parameters: Default
    Purpose: 
    '''
    mdls = [DecisionTreeRegressor(), DecisionTreeRegressor(criterion='poisson'), DecisionTreeRegressor(splitter='random')]
    print("\nDecision Trees\n-------------------")

    for model in mdls:
        print(f"Model: {model}")
        start = time.perf_counter_ns()
        predictions = cross_val_predict(model, X, y, cv=10)
        dt = time.perf_counter_ns() - start
        print(f"10-fold CV time: {dt / 1_000_000:.3f} ms")
        model.fit(X, y)
        print("Node count:", model.tree_.node_count)
        print("Depth:", model.tree_.max_depth)
        print("Leaf count:", model.tree_.n_leaves)
        reg_evaluation(predictions, y, model, save_graphs=save_graphs)

    # Random Forest - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    '''
    Library Function: RandomForestClassifier
    Parameters: 
    Purpose: 
    '''
    print("\nRandom Forests\n----------------")
    mdls = [RandomForestRegressor(), RandomForestRegressor(max_depth=5), RandomForestRegressor(n_estimators=50), RandomForestRegressor(criterion='poisson')]
    for model in mdls:
        print(f"Model: {model}")
        start = time.perf_counter_ns()
        predictions = cross_val_predict(model, X, y, cv=10)
        dt = time.perf_counter_ns() - start
        print(f"10-fold CV time: {dt / 1_000_000:.3f} ms")
        reg_evaluation(predictions, y, model, save_graphs=save_graphs)

    print("\nRegression Section Complete")
    return
    

def reg_evaluation(y_pred, y_true, model, *, verbose=True, graph=True, save_graphs=False):
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
        print(f"R^2 Score: {r2:.4f}\n")

    if graph:
        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.6)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel("Actual BMI")
        plt.ylabel("Predicted BMI")
        plt.title(f"Predicted vs Actual BMI, {model}")
        if save_graphs:
            FIGURES_DIR.mkdir(exist_ok=True)
            plt.savefig(FIGURES_DIR / f"regression_predicted_vs_actual_{model_filename(model)}.png", bbox_inches="tight")
        plt.show()

    return (correlation_coef, mae, mse, rmse, r2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-graphs", action="store_true")
    args = parser.parse_args()

    regression(save_graphs=args.save_graphs)
    classification(save_graphs=args.save_graphs)
    
if __name__ == "__main__":
    main()
