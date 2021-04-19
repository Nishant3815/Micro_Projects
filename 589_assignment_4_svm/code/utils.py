import csv
import numpy as np
from scipy.special import comb

def write_csv(y_pred, filename):
    """Write a 1d numpy array to a Kaggle-compatible .csv file"""
    with open(filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Id', 'Category'])
        for idx, y in enumerate(y_pred):
            csv_writer.writerow([idx, y])

def get_poly_expansion(P):
    def expand(X):
        tmp = [np.sqrt(comb(P,p))*X**p for p in range(P+1)]
        return np.vstack(tmp).T
    return expand
    
