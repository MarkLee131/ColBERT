import os
import pandas as pd
from tqdm import tqdm

def count_queries_cves(filepath):
    df = pd.read_csv(filepath)
    cve_list = df['cve'].unique()
    queries_list = df['desc_token'].unique()
    return cve_list, queries_list

def count_queries_cves_all(dir:str='/mnt/local/Baselines_Bugs/ColBERT/data/query_cve_data'):
    cve_list = []
    queries_list = []
    for filename in os.listdir(dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(dir, filename)
            cve_list_, queries_list_ = count_queries_cves(filepath)
            cve_list.extend(cve_list_)
            queries_list.extend(queries_list_)
    print(f'Number of CVEs: {len(cve_list)}')
    print(f'Number of queries: {len(queries_list)}')
    print('*'*50)
    print(f'Number of unique CVEs: {len(set(cve_list))}')
    print(f'Number of unique queries: {len(set(queries_list))}')
            
            
    return cve_list, queries_list

COLLECTION_DIR = '/mnt/local/Baselines_Bugs/ColBERT/data/cve_split/collection_data'
os.makedirs(COLLECTION_DIR, exist_ok=True)
def sample_collections(filepath, owner, repo):
    '''
    sample the collections to 5000: n patch and 5000-n for non-patch
    '''
    df = pd.read_csv(filepath)
    df_patch = df[df['label']==1]
    n_sample = min(5000-len(df_patch), len(df[df['label']==0]))
    df_non_patch = df[df['label']==0].sample(n=n_sample, random_state=3407)
    df_sample = pd.concat([df_patch, df_non_patch], axis=0)
    # df_sample.to_csv(os.path.join(COLLECTION_DIR, f'{owner}_{repo}.csv'), index=False)
    
    # commits,owner,repo,commit_id,cve,label
    
    # reset the pid from 0
    df_sample['pid'] = range(len(df_sample))
    df_sample = df_sample[['pid', 'commits', 'cve', 'owner', 'repo', 'commit_id', 'label']]
    df_sample.to_csv(os.path.join(COLLECTION_DIR, f'{owner}_{repo}.tsv'), sep='\t', index=False, header=False, encoding='utf-8')

def sample_collections_all(dir:str='/mnt/local/Baselines_Bugs/ColBERT/data/repo_data'):
    '''
    sample the collections for all repos
    '''
    for filename in tqdm(os.listdir(dir)):
        if filename.endswith('.csv'):
            filepath = os.path.join(dir, filename)
            owner, repo = filename.split('.')[0].split('_', 1)
            sample_collections(filepath, owner, repo)

from utils import reduce_mem_usage
def split_cve(filepath:str):
    df = pd.read_csv(filepath)
    reduce_mem_usage(df)
    queries_df = df[['desc_token', 'cve', 'owner', 'repo']].drop_duplicates()
    queries_df['desc_token'] = queries_df['desc_token'].str.replace('\t', ' ')
    
    repo_df = queries_df.groupby(['owner', 'repo'])
    
    for (owner, repo), group in tqdm(repo_df, desc=f"Processing {filepath}"):
        ## we need to add a column as qid
        group['qid'] = range(len(group))
        group = group[['qid', 'desc_token', 'cve', 'owner', 'repo']]
        # group.to_csv(os.path.join(QUERY_DIR, f'{owner}_{repo}.csv'), index=False)
        # save to tsv
        if os.path.exists(os.path.join(QUERY_DIR, f'{owner}_{repo}.tsv')):
            # we append to the existing file, and set the qid from the last one
            group['qid'] = group['qid'] + pd.read_csv(os.path.join(QUERY_DIR, f'{owner}_{repo}.tsv'), sep='\t', header=None, usecols=[0])['qid'].max() + 1
            group.to_csv(os.path.join(QUERY_DIR, f'{owner}_{repo}.tsv'), sep='\t', index=False, header=False, mode='a', encoding='utf-8')
        else:
            # we create a new file
            group.to_csv(os.path.join(QUERY_DIR, f'{owner}_{repo}.tsv'), sep='\t', index=False, header=False, encoding='utf-8')
    
    

QUERY_DIR = '/mnt/local/Baselines_Bugs/ColBERT/data/cve_split/query_data'
os.makedirs(QUERY_DIR, exist_ok=True)
def split_cve_all(root_dir:str='/mnt/local/Baselines_Bugs/ColBERT/data'):
    for filename in ['train_data.csv', 'validate_data.csv', 'test_data.csv']:
        filepath = os.path.join(root_dir, filename)
        split_cve(filepath)

if __name__ == '__main__':
    # cve_list, queries_list = count_queries_cves_all()
    # Number of CVEs: 4794
    # Number of queries: 4731
    # **************************************************
    # Number of unique CVEs: 4789
    
    # sample_collections_all()
    
    split_cve_all()