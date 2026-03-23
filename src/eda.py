import pandas as pd
from data_loader import load_data

def eda(df):
    print("======Basic Info=====")
    print(df.info())
    
    print("\n=====Missing Values=====")
    print(df.isnull().sum())
    
    print("\n====Emotional State Distribution====")
    print(df['emotional_state'].value_counts())
    
    print("\n====Intensity Distribution====")
    print(df['intensity'].value_counts())
    
    print("\n===Numerical Summary====")
    print(df.describe())
    
def text_analysis(df):
    print("\n====Text Length Analysis====")
    
    df['text_length'] = df['journal_text'].apply(lambda x: len(str(x).split()))
    
    print(df['text_length'].describe())
    
    print("\nShortest texts:\n", df.nsmallest(5, 'text_length')['journal_text'])
    print("\nLongest texts:\n", df.nlargest(5, 'text_length')['journal_text'])

if __name__=='__main__':
    train,_=load_data()
    
    eda(train)
    text_analysis(train)
    
    