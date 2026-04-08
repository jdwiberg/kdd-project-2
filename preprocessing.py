import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

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

def regression_pp(*, include_stroke = True): # use BMI for regression
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def classification_pp(*, upsample = False): # use stroke for classification
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if upsample:
        # upsample the minority class in the training split only
        X_maj = X_train[y_train == 0]
        y_maj = y_train[y_train == 0]
        X_min = X_train[y_train == 1]
        y_min = y_train[y_train == 1]

        X_min_upsampled, y_min_upsampled = resample(
            X_min,
            y_min,
            replace=True,  # Sample with replacement
            n_samples=len(y_maj),  # Match number of maj samples
            random_state=42,
        )

        # combine the original majority class with the upsampled minority class
        X_train = pd.concat([X_maj, X_min_upsampled])
        y_train = pd.concat([y_maj, y_min_upsampled])

    return X_train, X_test, y_train, y_test
