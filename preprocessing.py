import kagglehub
from kagglehub import KaggleDatasetAdapter

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

def regression_pp(): # use BMI for regression
    # drop rows with missing BMI values, since we can't use them for regression
    X = dataframe.copy().dropna(subset=['bmi'])
    y = X.pop('bmi', axis=1)
    
    return (X, y)

def classification_pp(): # use stroke for classification
    # drop rows with missing stroke values, since we can't use them for regression
    X = dataframe.copy().dropna(subset=['stroke'])
    y = X.pop('stroke')

    return (X, y)