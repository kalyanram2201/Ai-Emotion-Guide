import pandas as pd

def load_data():
    train = pd.read_csv("data/Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv")
    test = pd.read_csv("data/arvyax_test_inputs_120.xlsx - Sheet1.csv")
    return train, test

if __name__ == '__main__':
    train, test = load_data()
    
    print("Train Shape:", train.shape)
    print("Test Shape:", test.shape)
    
    print("\nColumns:\n", train.columns)
    
    print("\nSample Data:\n", train.head())