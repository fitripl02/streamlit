import pandas as pd

def load_data():
    df = pd.read_csv("semarang_resto_dataset.csv")
    df.dropna(inplace=True)
    return df
