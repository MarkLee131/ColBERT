import os
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename='verify_format.log', filemode='a')
def verify_format(input_filename):
### verify the tsv file has the correct format
    # df = pd.read_csv(input_filename, sep='\t')
    # print(df.head())
    # print(df.shape)
    # print(df.columns)
    # print(df.dtypes)
    # #### we need to verify that whether the columns are three, and the data type is string, and split by \t
    # if len(df.columns) != 3:
    #     print('The column number is not correct')
    #     return False
    # if df.dtypes[0] != 'string' or df.dtypes[1] != 'string' or df.dtypes[2] != 'string':
    #     print('The data type is not correct')
    #     return False
    reader = open(input_filename, mode='r', encoding="utf-8")
    
    for line_idx, line in tqdm(enumerate(reader)):
        try:
            _, _, _ = line.strip().split('\t')
        except:
            print(f'Invalid line: {line}')
            continue
    

def find_empty_cell(input_filename):
    df = pd.read_csv(input_filename, sep='\t')
    print(df.head())
    print(df.shape)
    print(df.columns)
    print(df.dtypes)
    print(df.iloc[3227919,1])
    print("-"*20)
    #### we need to verify that whether the columns are three, and the data type is string, and split by \t
    if len(df.columns) != 3:
        print('The column number is not correct')
        return False
    if df.dtypes[0] != 'string' or df.dtypes[1] != 'string' or df.dtypes[2] != 'string':
        print('The data type is not correct')
        return False
    for i in range(df.shape[0], start=1):
        for j in range(df.shape[1], start=3227919):
            if df.iloc[i, j] == '':
                print(f'Empty cell: {i} {j}')
                return False
    return True


if __name__ == "__main__":
    test_file = './data/train_data.tsv'
    
    # verify_format(test_file)
    find_empty_cell(test_file)
    
    