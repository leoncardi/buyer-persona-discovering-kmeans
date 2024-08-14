import pandas as pd

def dropna(data):
    data = data.dropna()
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    return data

def drop_columns(data, columns):
    data = data.drop(columns, axis=1)
    return data

def main():
    data = pd.read_csv('data/raw_data.csv')
    data = dropna(data)
    data = drop_columns(data, ['CustomerID'])
    data.to_csv('data/preprocessed.csv', index=False)

if __name__ == '__main__':
    main()
