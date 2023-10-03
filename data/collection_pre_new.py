import datetime
import gc
import logging
import os
import pandas as pd
from data_prepare import reduce_mem_usage

'''
find the cve commit id from the original train_data.csv, validate_data.csv, test_data.csv
'''

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    filename=f'collection_pre{datetime.date.today()}.log', filemode='w')

def get_collection(file_path, save_suffix):
    try:
        logging.info(f"Processing file: {file_path}")
        
        df = pd.read_csv(file_path)
        reduce_mem_usage(df)
        
        logging.info("Memory usage reduced")
        
        df['desc_token'] = df['desc_token'].str.replace('\t', ' ', regex=False)
        desc_token = df[['cve', 'desc_token']].drop_duplicates(subset=['desc_token', 'cve']).reset_index(drop=True)
        desc_token['qid'] = range(0, len(desc_token))
        
        # Save the mapping of qid to cve and commit_id
        qid_mapping = desc_token[['qid', 'cve', 'commit_id']]
        print(qid_mapping.shape)
        
        qid_mapping.to_csv(os.path.join(os.path.dirname(file_path), f'qid_mapping_{save_suffix}.csv'), index=False)
        
        # desc_token = desc_token[['qid', 'desc_token']]
        # logging.info(f"desc_token processed with columns: {desc_token.columns}")

                
        # desc_token.to_csv(os.path.join(os.path.dirname(file_path), f'queries_{save_suffix}.tsv'), sep='\t', index=False, header=False)
        # logging.info("queries.tsv saved")
        
        
        df['combined'] = df['msg_token'].str.cat(df['diff_token'], sep=" ").str.replace('\t', ' ', regex=False)
        combined = df[['cve', 'commit_id', 'combined']].drop_duplicates(subset=['combined']).reset_index(drop=True)
        combined['cid'] = range(0, len(combined))
        
        # Save the mapping of cid to cve and commit_id
        cid_mapping = combined[['cid', 'cve', 'commit_id']]
        cid_mapping.to_csv(os.path.join(os.path.dirname(file_path), f'cid_mapping_{save_suffix}.csv'), index=False)
        
        combined = combined[['cid', 'combined']]
        logging.info(f"combined column processed with columns: {combined.columns}")
        
        del df['msg_token']
        del df['diff_token']
        
        logging.info("msg_token and diff_token columns deleted")
        
        combined.to_csv(os.path.join(os.path.dirname(file_path), f'collection_{save_suffix}.tsv'), sep='\t', index=False, header=False)
        logging.info("collection.tsv saved")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")



BASE_PATH = '/mnt/local/Baselines_Bugs/ColBERT/data'

def merge_files(quries_files, collection_files, base_path=BASE_PATH):
    
    # Process queries files
    df_list = []
    for file in quries_files:
        file_path = os.path.join(base_path, file)
        logging.info(f"Starting to process query file: {file_path}")
        
        current_df = pd.read_csv(file_path, sep='\t', names=['qid', 'desc_token'])
        df_list.append(current_df)
        
        logging.info(f"Processed {file_path}. Shape: {current_df.shape}")
        
    queries = pd.concat(df_list, ignore_index=True)
    logging.info(f"Combined all queries files. Shape before deduplication: {queries.shape}")
    
    queries = queries.drop_duplicates(subset=['desc_token']).reset_index(drop=True)
    logging.info(f"Shape after deduplication: {queries.shape}")

    queries['qid'] = range(0, len(queries))
    queries.to_csv(os.path.join(base_path, 'queries_all.tsv'), sep='\t', index=False, header=False)
    logging.info(f"Saved combined queries to queries_all.tsv")
    
    # Process collection files
    df_list = []
    for file in collection_files:
        file_path = os.path.join(base_path, file)
        logging.info(f"Starting to process collection file: {file_path}")
        
        current_df = pd.read_csv(file_path, sep='\t', names=['cid', 'commits_token'])
        df_list.append(current_df)
        
        logging.info(f"Processed {file_path}. Shape: {current_df.shape}")
    
    collection = pd.concat(df_list, ignore_index=True)
    logging.info(f"Combined all collection files. Shape before deduplication: {collection.shape}")
    
    collection = collection.drop_duplicates(subset=['commits_token']).reset_index(drop=True)
    logging.info(f"Shape after deduplication: {collection.shape}")

    collection['cid'] = range(0, len(collection))
    collection.to_csv(os.path.join(base_path, 'collection_all.tsv'), sep='\t', index=False, header=False)
    logging.info(f"Saved combined collection to collection_all.tsv")

    
    
if __name__ == "__main__":
    train_path = '/mnt/local/Baselines_Bugs/ColBERT/data/train_data.csv'
    validate_path = '/mnt/local/Baselines_Bugs/ColBERT/data/validate_data.csv'
    test_path = '/mnt/local/Baselines_Bugs/ColBERT/data/test_data.csv'
    
    run_collection = True
    # run_merge = True

    if run_collection:
        # get_collection(train_path, 'train')
        # get_collection(validate_path, 'validate')
        get_collection(test_path, 'test')

    # if run_merge:
    #     merge_files(['queries_train.tsv', 'queries_validate.tsv', 'queries_test.tsv'],
    #                 ['collection_train.tsv', 'collection_validate.tsv', 'collection_test.tsv'])
