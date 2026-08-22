import pandas as pd
import numpy as np

def load_data(uploaded_file):
    """Loads an uploaded CSV file safely into a Pandas DataFrame."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {e}")

def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Computes basic health stats for the dataset."""
    return {
        "total_rows": int(df.shape[0]),
        "total_cols": int(df.shape[1]),
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
    }

def search_dataset(df: pd.DataFrame, search_query: str) -> pd.DataFrame:
    """Filters dataset rows matching a text search query."""
    if not search_query.strip():
        return df
    mask = np.column_stack([
        df[col].astype(str).str.contains(search_query, case=False, na=False) 
        for col in df.columns
    ])
    return df.loc[mask.any(axis=1)]