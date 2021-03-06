import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer, MinMaxScaler

'''Useful functions and methods'''
class processing:

    def __init__(self):
        pass
    ''' This function loads a csv file to a pandas data frame'''
    def read_csv(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            print("file read as csv")
            return df

        except FileNotFoundError:
            print("file not found")
    ''' This function saves a csv file to a pandas data frame'''
    def save_csv(self, df, csv_path):
        try:
            df.to_csv(csv_path, index=False)
            print('File Successfully Saved.!!!')

        except Exception:
            print("Save failed...")

        return df
    '''The percent_missing function counts the number of missing values in in percentage'''
    def percent_missing(self, df: pd.DataFrame) -> float:

        totalCells = np.product(df.shape)
        missingCount = df.isnull().sum()
        totalMissing = missingCount.sum()
        return round((totalMissing / totalCells) * 100, 2)
    '''The percent_missing function counts the number of missing columns in in percentage'''
    
    def percent_missing_for_col(self, df: pd.DataFrame, col_name: str) -> float:
        total_count = len(df[col_name])
        if total_count <= 0:
            return 0.0
        missing_count = df[col_name].isnull().sum()

        return round((missing_count / total_count) * 100, 2)


    #  for pharmaceutical
    def ToWeight(y):
        w = np.zeros(y.shape, dtype=float)
        ind = y != 0
        w[ind] = 1./(y[ind]**2)
        return w

    def rmspe(yhat, y):
        w = ToWeight(y)
        rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
        return rmspe

    def rmspe_xg(yhat, y):
        y = y.get_label()
        y = np.exp(y) - 1
        yhat = np.exp(yhat) - 1
        w = ToWeight(y)
        rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
        return  rmspe