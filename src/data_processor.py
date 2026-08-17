import pandas as pd
import numpy as np

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes memory usage of a Pandas DataFrame by downcasting numeric types.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / 1024 ** 2
    return df

def get_dataset_stats(df: pd.DataFrame) -> dict:
    """
    Computes summary health statistics for the dataset.
    """
    stats = {
        "total_rows": int(df.shape[0]),
        "total_cols": int(df.shape[1]),
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
    }
    return stats

def get_column_recommendations(df: pd.DataFrame) -> dict:
    """
    Generates basic cleaning/type recommendations for columns.
    """
    recommendations = {}
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            recommendations[col] = f"Contains {missing} missing values. Consider imputation."
        else:
            recommendations[col] = "Clean"
    return recommendations