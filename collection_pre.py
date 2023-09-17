import gc
import os
import pandas as pd
from data.data_prepare import reduce_mem_usage
import datetime
import logging

today = datetime.date.today()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=f'collection_pre{today}.log', filemode='w')

def get_collection(file_path):
    
    try:
        print(f"{datetime.datetime.now()} - Processing file: {file_path}")
        # Read the file
        df = pd.read_csv(file_path)
        
        # Reduce memory usage
        reduce_mem_usage(df)
        # print(f"{datetime.datetime.now()} - Memory usage reduced")
        logging.info(f"{datetime.datetime.now()} - Memory usage reduced")
        
        # Process desc_token column
        df['desc_token'] = df['desc_token'].str.replace('\t', ' ', regex=False)
        desc_token = df['desc_token'].drop_duplicates().reset_index(drop=True)
        desc_token = pd.DataFrame({'qid': range(1, len(desc_token) + 1), 'desc_token': desc_token})
        # print(f"Columns: {desc_token.columns}")
        logging.info(f"Columns: {desc_token.columns}")
        logging.info(f"{desc_token.shape}")
        
        # print(f"{datetime.datetime.now()} - desc_token processed")
        logging.info(f"{datetime.datetime.now()} - desc_token processed")
        
        desc_token.to_csv(os.path.join(os.path.dirname(file_path), 'queries.tsv'), sep='\t', index=False, header=False)
        # print(f"{datetime.datetime.now()} - queries.tsv saved")
        logging.info(f"{datetime.datetime.now()} - queries.tsv saved")
        
        # Process msg_token and diff_token columns
        df['combined'] = df['msg_token'] + " " + df['diff_token']
        df['combined'] = df['combined'].str.replace('\t', ' ', regex=False)
        
        # print(f"{datetime.datetime.now()} - combined column created")
        logging.info(f"{datetime.datetime.now()} - combined column created")
        
        combined = df['combined'].drop_duplicates().reset_index(drop=True)
        combined = pd.DataFrame({'cid': range(1, len(combined) + 1), 'commits_token': combined})
        # print(f"{datetime.datetime.now()} - combined column processed")
        # print(f"Columns: {combined.columns}")
        
        logging.info(f"{datetime.datetime.now()} - combined column processed")
        logging.info(f"Columns: {combined.columns}")
        logging.info(f"{combined.shape}")
        
        
        del df['msg_token']
        del df['diff_token']
        gc.collect()
        
        # print(f"{datetime.datetime.now()} - msg_token and diff_token columns deleted")
        logging.info(f"{datetime.datetime.now()} - msg_token and diff_token columns deleted")
        
        combined.to_csv(os.path.join(os.path.dirname(file_path), 'collection.tsv'), sep='\t', index=False, header=False)
        
        # print(f"{datetime.datetime.now()} - collection.tsv saved")
        logging.info(f"{datetime.datetime.now()} - collection.tsv saved")
        
    except Exception as e:
        # print(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    file_path = '/mnt/local/Baselines_Bugs/ColBERT/data/train_data.csv'
    get_collection(file_path)
