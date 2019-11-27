import csv
import numpy as np


def load(filename):
    """
    Loads a CSV file matching the format of the Kaggle dataset at https://www.kaggle.com/mlg-ulb/creditcardfraud.
    Returns a 2D numpy array containing all the data, except for the time, and the class column is cast to a float.
    """
    print('Loading "{}"...'.format(filename))
    data = []
    with open(filename, newline='\n') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)
        for row in reader:
            # Get rid of the time column.
            row = row[1:]
            # Cast the class to a float.
            row[-1] = float(row[-1])
            data.append(row)
    return np.array(data, dtype=np.float32)
