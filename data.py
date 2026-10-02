import pandas as pd


def load_data(path):
    """
    Load the retail demand dataset.
    """

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    df = df.drop_duplicates()

    return df