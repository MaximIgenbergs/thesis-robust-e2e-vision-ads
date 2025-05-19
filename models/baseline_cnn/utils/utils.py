import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataframes(csv_path, val_split=0.2, random_seed=42):
    """
    Reads the driving log CSV and splits into train/validation DataFrames.
    Returns:
      train_df, val_df
    """
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=random_seed
    )
    return train_df, val_df
