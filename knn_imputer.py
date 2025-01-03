import copy
import numpy.ma as ma
from tqdm import tqdm 
from constants import *
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Utility function to cast array elements to integers
cast_int = lambda i: int(i)
castFunction = np.vectorize(cast_int)


def find_label(data, labels):
    """
    Maps each value in `data` to the nearest label in `labels`.

    Parameters:
    data (array-like): Input data to be mapped.
    labels (list or array-like): List of possible labels.

    Returns:
    list: New data with values replaced by the nearest labels.
    """
    selected = 0
    new_data = []
    for i in data:
        min_distance = float('inf')
        for label in labels:
            # Compute the distance between the current value and the label
            distance = abs(int(i) - label)
            if distance < min_distance:
                selected = label
                min_distance = distance
        new_data.append(selected)
    return new_data


def tune_knn(x_fold, columns, rng=(1, 31), n_splits=32, shuffle=True, random_state=2021, average='binary'):
    """
    Tunes the number of neighbors (k) for KNN Imputer using K-Fold cross-validation.

    Parameters:
    x_fold (array-like): Input dataset with missing values.
    columns (list): List of columns to perform imputation on.
    rng (tuple): Range of neighbors (inclusive) to evaluate. Default is (1, 31).
    n_splits (int): Number of folds for cross-validation. Default is 32.
    shuffle (bool): Whether to shuffle data before splitting. Default is True.
    random_state (int): Seed for reproducibility. Default is 2021.
    average (str): Metric averaging strategy. Default is 'binary'.

    Returns:
    list: Mean squared errors for each number of neighbors evaluated.
    """
    # Scale the dataset to normalize values
    scaler = MinMaxScaler()

    # Handle missing values in the dataset
    accuracy = []
    y_fold = copy.deepcopy(x_fold)
    # Fill missing values with the column-wise mean
    y_fold = np.where(np.isnan(y_fold), ma.array(y_fold, mask=np.isnan(y_fold)).mean(axis=0), y_fold)
    # Cast filled values to integers
    y_fold = castFunction(y_fold)

    for neighbors in tqdm(range(rng[0], rng[1]), desc="Processing"):
        algo = KNNImputer(n_neighbors=neighbors)  # Initialize KNN Imputer
        temp = []

        # Perform K-Fold cross-validation
        cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for column in columns:
            for train_index, test_index in cv.split(x_fold):
                x_train = copy.deepcopy(x_fold)
                x_train = scaler.fit_transform(x_train)  # Scale training data
                x_train[test_index, [column]] = np.nan  # Set test set column values to NaN
                x_train = algo.fit_transform(x_train)  # Perform KNN Imputation
                x_train = scaler.inverse_transform(x_train)  # Reverse scaling
                x_train = castFunction(x_train)  # Cast imputed values to integers

                # Compute the mean squared error for imputed values
                temp.append(mean_squared_error(x_train[test_index, [column]], y_fold[test_index, [column]]))

        accuracy.append(np.array(temp).mean())

    return accuracy




class Imputer:
    """
    A class for handling missing values using KNN Imputer and performing data cleanup.

    Attributes:
    df_init (pd.DataFrame): Original dataset with missing values.
    df (pd.DataFrame): Dataset after imputation.
    """
    def __init__(self, df, k=15):
        """
        Initializes the Imputer with a dataset and performs KNN Imputation.

        Parameters:
        df (pd.DataFrame): Input dataset with missing values.
        k (int): Number of neighbors for KNN Imputer. Default is 15.
        """
        self.df_init = df
        self.df = df.copy()
        imputer = KNNImputer(n_neighbors=k)
        # Perform KNN Imputation
        self.df[self.df.columns] = imputer.fit_transform(self.df)

    def nearest_value(self, x, uniqueValues):
        """
        Finds the nearest value to `x` from the set of unique values.

        Parameters:
        x (float): Value to find the nearest match for.
        uniqueValues (array): Array of unique values to compare against.

        Returns:
        float: Nearest value from `uniqueValues`.
        """
        diff = np.abs(uniqueValues - x)
        return uniqueValues[np.argmin(diff)]

    def clean(self, missValCols, dataTypeDict):
        """
        Cleans imputed columns based on their data types.

        Parameters:
        missValCols (dict): Dictionary mapping columns to their count of missing values.
        dataTypeDict (dict): Dictionary mapping columns to their data types.
        
        Modifies:
        - Columns of the DataFrame based on their data type (e.g., rounding, category matching).
        """
        for col in missValCols.keys():
            if missValCols[col] > 0:
                dataType = dataTypeDict[col]
                if dataType == 'category' or dataType == 'binary':
                    # Map imputed values to the nearest unique value in the original dataset
                    uniqueValues = np.array(self.df_init[col].unique()[:-1])
                    self.df[col] = self.df[col].apply(lambda x: self.nearest_value(x, uniqueValues))