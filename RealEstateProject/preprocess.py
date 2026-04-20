# STEP 2: Data Preprocessing (src/preprocess.py)
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df = df.drop_duplicates()

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Feature Engineering
    df['Price_per_SqFt'] = df['Price_in_Lakhs'] * 100000 / df['Size_in_SqFt']
    df['Age_of_Property'] = 2026 - df['Year_Built']

    return df