import copy
import numpy.ma as ma
from tqdm import tqdm 
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