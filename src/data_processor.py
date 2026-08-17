import pandas as pd
import numpy as np

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes the memory footprint of a Pandas DataFrame by downcasting numeric types.
    This prevents memory exhaustion on cloud servers (Streamlit Cloud).
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and not pd.api.types.is_datetime64_any_dtype(df[col]):
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
    print(f"Memory optimization: Reduced from {start_mem:.2f} MB to {end_mem:.2f} MB")
    return df

def load_and_process_data(uploaded_file, max_rows=100000):
    """
    Loads an uploaded CSV file safely, limits row counts for performance, 
    and applies memory optimization.
    """
    try:
        # Read file with row limit safety check
        df = pd.read_csv(uploaded_file, nrows=max_rows)
        
        # Optimize memory usage
        df = optimize_memory(df)
        return df
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {e}")

def get_data_health_metrics(df: pd.DataFrame) -> dict:
    """Computes core data quality metrics for KPI cards."""
    total_rows = df.shape[0]
    total_cols = df.shape[1]
    missing_cells = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    
    total_cells = total_rows * total_cols
    missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0.0
    memory_usage_mb = round(df.memory_usage().sum() / 1024 ** 2, 2)

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "missing_cells": missing_cells,
        "missing_percentage": round(missing_percentage, 2),
        "duplicate_rows": duplicate_rows,
        "memory_usage_mb": memory_usage_mb
    }

def get_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Returns data types and missing counts per column for deep inspection."""
    summary_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })
    return summary_df