import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime



def process_csv_to_tsv(input_filename, output_filename):
    '''
    get the triplets tsv from the train_data.csv
    '''
    
    start = datetime.now()
    
    print(start)
    print(f"Processing {input_filename} to {output_filename}")
    
    df = pd.read_csv(input_filename)
    
    # Reduce memory usage
    reduce_mem_usage(df)
    
    # Add a unique identifier column to trace back to the original row
    df['unique_id'] = df.index
    
    # List to store the new rows
    new_rows = []
    
    columns2check = ['desc_token', 'msg_token', 'diff_token']
    # Replace or escape any \t characters in string columns
    for col in columns2check:
        if df[col].dtype == 'object':
            print(f"Processing {col} as object")
            df[col] = df[col].str.replace('\t', ' ')
        elif df[col].dtype == str:
            print(f"Processing {col} as str")
            df[col] = df[col].replace('\t', ' ')
        else:
            print(f"Skipping {col}")
    
    
    # Group by CVE and process each group
    for cve, group in tqdm(df.groupby('cve'), desc=f"Processing {input_filename}"):
        desc_token = group['desc_token'].iloc[0]
        pos_rows = group[group['label'] == 1]
        neg_rows = group[group['label'] == 0]
        # neg_rows = neg_rows[:min(len(neg_rows), 5000-len(pos_rows))]        
        neg_rows['combined'] = neg_rows['msg_token'] + " " + neg_rows['diff_token']
        
        for _, pos_row in pos_rows.iterrows():
            pos_info = pos_row['msg_token'] + " " + pos_row['diff_token']
            
            for _, neg_row in neg_rows.iterrows():
                if neg_row['combined'] != neg_row['combined']:
                    print(f"Skipping NaN row: {neg_row}")
                    continue
                elif neg_row['combined'] == '':
                    print(f"Skipping empty row: {neg_row}")
                    continue
                
                new_rows.append({
                    'desc_token': desc_token,
                    'pos_info': pos_info,
                    'neg_info': neg_row['combined'],
                    'pos_unique_id': pos_row['unique_id'],
                    'neg_unique_id': neg_row['unique_id']
                })
    
    # Convert the list of new rows to a DataFrame
    del df
    new_df = pd.DataFrame(new_rows)
    
    reduce_mem_usage(new_df)

    finish = datetime.now()
    duration = finish - start
    ### display the time taken to process the file in hours, minutes and seconds
    print(f"Finished processing {input_filename} to {output_filename} in {duration}")
    
    print(f"Writing to {output_filename}...")
    # Write to TSV
    new_df.to_csv(output_filename, sep='\t', index=False)


            
def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

if __name__ == '__main__':
    import os
    # Call the function for each of the three files
    save_dir = '/mnt/local/Baselines_Bugs/ColBERT/data/'
    # process_csv_to_tsv('test_data.csv', 'test_data.tsv')
    # process_csv_to_tsv('validate_data.csv', 'validate_data.tsv')
    process_csv_to_tsv(os.path.join(save_dir, 'train_data.csv'), os.path.join(save_dir, 'train_data_triplets.tsv'))
    
''' 
CUDA_VISIBLE_DEVICES="0,1,2,3" \
python -m torch.distributed.launch --nproc_per_node=4 -m \
colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 256 --accum 1 \
--triples data/train_data.tsv \
--root commits_exp --experiment commits_train --similarity l2 --run test.l2
'''
