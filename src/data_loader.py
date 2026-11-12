import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_data():
    """
    Load California Housing dataset
    and return it as a Pandas DataFrame.
    """

    housing = fetch_california_housing(as_frame=True)

    df = housing.frame.copy()

    return df


if __name__ == "__main__":
    df = load_data()

    print("Dataset loaded successfully!")
    print("\nFirst 5 records:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())
