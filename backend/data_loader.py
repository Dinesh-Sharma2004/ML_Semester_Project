
import os
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(BASE_DIR, "data", "dataset_train.csv")
METRO_PATH = os.path.join(BASE_DIR, "data", "MetroPT2.csv")


def load_optimized_dataset(name: str = "train", nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Loads dataset optimized for memory.
    1. Reads only sensor columns + key time cols using nrows.
    2. Downcasts all sensor data to float32.
    """
    if name == "train":
        path = TRAIN_PATH
    elif name == "metro":
        path = METRO_PATH
    else:
        raise ValueError("Dataset name must be 'train' or 'metro'")

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Place the dataset inside /backend/data/")

    try:

        all_cols = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception as e:
        raise FileNotFoundError(f"Could not read header of {path}. Error: {e}")

    exclude = set(["id", "trip_id", "timestamp", "time", "date", "label"])
    sensors = [c for c in all_cols if c not in exclude and not c.lower().startswith('unnamed')]
    dtype_map = {col: 'float32' for col in sensors}
    time_cols = ["timestamp", "time", "date"]
    use_cols = sensors + [col for col in time_cols if col in all_cols]

    try:

        df = pd.read_csv(
            path,
            usecols=use_cols,
            dtype=dtype_map,
            nrows=nrows, 
            low_memory=False
        )
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not load with float32, falling back. Error: {e}")
        df = pd.read_csv(
            path,
            usecols=use_cols,
            nrows=nrows, 
            low_memory=False
        )
    except MemoryError:
         raise MemoryError(f"Still ran out of memory reading {nrows if nrows else 'all'} rows from {path}. Try reducing nrows further or check system RAM / Python bitness (should be 64-bit).")
    except Exception as e:
         raise RuntimeError(f"An unexpected error occurred reading {path}: {e}")

    for col in time_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df = df.sort_values(col)
            except Exception as date_err:
                 print(f"Warning: Could not parse or sort by date column {col}. Error: {date_err}")
            break

    time_col_found = next((col for col in time_cols if col in df.columns), None)
    if time_col_found:
        initial_rows = len(df)
        df.dropna(subset=[time_col_found], inplace=True)
        if len(df) < initial_rows:
            print(f"Dropped {initial_rows - len(df)} rows due to invalid date/time entries.")

    return df

def select_sensor_columns(df: pd.DataFrame) -> List[str]:
    exclude = set(["id", "trip_id", "timestamp", "time", "date", "label"])
    return [c for c in df.columns if df[c].dtype == 'float32' and c not in exclude]

def extract_window_features(df: pd.DataFrame, sensors: List[str], window: int = 5) -> pd.DataFrame:
    if not sensors:
        raise ValueError("No valid sensor columns found to calculate features.")
    df2 = pd.DataFrame()
    df2["row_mean"] = df[sensors].mean(axis=1).astype('float32')
    df2["roll_mean"] = df2["row_mean"].rolling(window=window, min_periods=1).mean()
    df2["roll_var"] = df2["row_mean"].rolling(window=window, min_periods=1).var().fillna(0)
    df2["trend"] = df2["row_mean"].diff(periods=window-1).fillna(0)
    df2["target"] = df2["row_mean"].shift(-1)
    initial_rows = len(df2)
    df2 = df2.dropna(subset=["roll_mean", "roll_var", "trend", "target"]).reset_index(drop=True)
    if len(df2) < initial_rows:
        print(f"Dropped {initial_rows - len(df2)} rows due to NaNs created during feature engineering.")
    return df2[["roll_mean", "roll_var", "trend", "target"]]

def prepare_training_data(name: str = "train", window: int = 5, max_rows: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Main function to load and process data, returning NumPy arrays.
    Uses max_rows to limit initial read via nrows.
    """
    df = load_optimized_dataset(name, nrows=max_rows) 
    sensors = select_sensor_columns(df)

    feats = extract_window_features(df, sensors, window)

    if len(feats) == 0:
        raise ValueError(f"No valid data remaining after feature engineering for dataset '{name}'.")
    X = feats[["roll_mean", "roll_var", "trend"]].values.astype('float32')
    y = feats["target"].values.astype('float32')

    meta = {
        "dataset": name,
        "rows": int(len(feats)), 
        "sensors": sensors
    }
    return X, y, meta