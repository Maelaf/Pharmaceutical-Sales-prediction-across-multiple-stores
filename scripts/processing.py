import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer, MinMaxScaler


class processing:

    def __init__(self):
        pass
    
    def read_csv(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            print("file read as csv")
            return df

        except FileNotFoundError:
            print("file not found")
    
    def save_csv(self, df, csv_path):
        try:
            df.to_csv(csv_path, index=False)
            print('File Successfully Saved.!!!')

        except Exception:
            print("Save failed...")

        return df
        
    def percent_missing(self, df: pd.DataFrame) -> float:

        totalCells = np.product(df.shape)
        missingCount = df.isnull().sum()
        totalMissing = missingCount.sum()
        return round((totalMissing / totalCells) * 100, 2)
    
    def percent_missing_for_col(self, df: pd.DataFrame, col_name: str) -> float:
        total_count = len(df[col_name])
        if total_count <= 0:
            return 0.0
        missing_count = df[col_name].isnull().sum()

        return round((missing_count / total_count) * 100, 2)
