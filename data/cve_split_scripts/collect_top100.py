'''
collect top100 from colbert, for split cve data
2023.10.07
'''
import gc
import os
import pandas as pd
from tqdm import tqdm
from utils import reduce_mem_usage

BASE_DIR = '/mnt/local/Baselines_Bugs/ColBERT/data'
QUERY_DIR = '/mnt/local/Baselines_Bugs/ColBERT/data/cve_split/query_data'
COLLECTION_DIR = '/mnt/local/Baselines_Bugs/ColBERT/data/cve_split/collection_data'
RANKING_DIR = '/mnt/local/Baselines_Bugs/ColBERT/data/cve_split/ranking_data'

TOP100_DIR = '/mnt/local/Baselines_Bugs/ColBERT/data/cve_split/top100_data'
os.makedirs(TOP100_DIR, exist_ok=True)

TRAIN_DATA = '/mnt/local/Baselines_Bugs/ColBERT/data/train_data.csv'
TEST_DATA = '/mnt/local/Baselines_Bugs/ColBERT/data/test_data.csv'
VALIDATION_DATA = '/mnt/local/Baselines_Bugs/ColBERT/data/validate_data.csv'

QUERY_CVE_DIR = '/mnt/local/Baselines_Bugs/ColBERT/data/query_cve_data'

def collect_top100_all(
    query_dir=QUERY_DIR,
    collection_dir=COLLECTION_DIR,
    top100_dir=TOP100_DIR,
    ranking_dir=RANKING_DIR,
    ):
    '''
    collect top100 from colbert by call collect_top100()
    '''
    # search for all query files ending with .tsv
    # repos = [file.split('.')[0] for file in os.listdir(collection_dir) if file.endswith('.tsv')]
    query_files = [os.path.join(query_dir, file) for file in os.listdir(query_dir) if file.endswith('.tsv')]
    print('num of query files:', len(query_files))
    
    
    for query_file in tqdm(query_files, total=len(query_files)):
        # get repo name from query file name
        filename = query_file.split('/')[-1]
        # get collection file name, same as the query file name
        collection_file = os.path.join(collection_dir, filename)
        
        # get ranking file name by searching the filename in ranking_dir `ranking.tsv`
        repo_name = filename.rsplit('.', 1)[0]
        ranking_file = os.path.join(ranking_dir, repo_name+'.tsv')
        
        top100_file = os.path.join(top100_dir, repo_name+'.csv')
        # check
        # if os.path.exists(top100_file):
        #     print('top100 file exists:', top100_file)
        #     continue
        collect_top100(
            query_file=query_file,
            collection_file=collection_file,
            ranking_file=ranking_file,
            top100_file=top100_file,
            repo_name=repo_name
            )
        # break


def collect_top100(
    query_file,
    collection_file,
    ranking_file,
    top100_file,
    repo_name
    ):
    '''
    collect top100 from colbert
    '''
    
    queries = pd.read_csv(query_file, sep='\t', header=None)
    queries.columns = ['qid', 'desc_token', 'cve', 'owner', 'repo']
    
    rankings = pd.read_csv(ranking_file, sep='\t', header=None)
    rankings.columns = ['qid', 'pid', 'rank']
    
    rankings = rankings.merge(queries, on='qid', how='left')
    # now the ranking has columns: qid, pid, rank, desc_token
    
    collections = pd.read_csv(collection_file, sep='\t', header=None)
    # reduce_mem_usage(collections)
    collections.columns = ['pid', 'commits', 'cve', 'owner', 'repo', 'commit_id', 'label']
    collections = collections.drop(['cve', 'label'], axis=1)
    
    rankings = rankings.merge(collections, on=['pid','owner','repo'], how='left')
    # now the ranking has columns: qid, pid, rank, desc_token, owner, repo, commits, commit_id
    
    
    ###  new: merge cve back based on desc_token
    query_cve_df = pd.read_csv(os.path.join(QUERY_CVE_DIR, repo_name+'.csv'))
    reduce_mem_usage(query_cve_df)
    # now query_cve_df has columns: desc_token, cve, commit_id, label
    query_cve_df = query_cve_df.drop(['desc_token',], axis=1)
    
    rankings = rankings.merge(query_cve_df, on=['cve','commit_id'], how='left')
    ## pad nan with 0 for label
    rankings['label'] = rankings['label'].fillna(0)
    
    rankings.to_csv(top100_file, index=False)
    print('save top100 to {}'.format(top100_file))

    
    

###  new: group (cve, desc_token, commit_id, label) by (owner, repo)
def split_data_by_repo():
    '''
    split data by repo, and save to QUERY_CVE_DIR
    '''
    # output_dir = '/mnt/local/Baselines_Bugs/ColBERT/data/query_cve_data'
    os.makedirs(QUERY_CVE_DIR, exist_ok=True)
    
    train_df = pd.read_csv(TRAIN_DATA)
    reduce_mem_usage(train_df)
    test_df = pd.read_csv(TEST_DATA)
    reduce_mem_usage(test_df)
    validate_df = pd.read_csv(VALIDATION_DATA)
    reduce_mem_usage(validate_df)
    merged_df = pd.concat([train_df, test_df, validate_df], ignore_index=True)
    del train_df, test_df, validate_df
    gc.collect()
    # # reduce_mem_usage(merged_df)
    # # merged_df = merged_df.drop(['owner', 'repo', 'msg_token', 'diff_token'], axis=1)
    merged_df = merged_df[['desc_token', 'cve', 'commit_id', 'label','repo','owner']]
    reduce_mem_usage(merged_df)
    
    # get all repo groups
    repo_groups = merged_df.groupby(['repo','owner'])
    for (repo,owner), repo_df in tqdm(repo_groups, total=len(repo_groups)):
        repo_df = repo_df.drop(['repo','owner'], axis=1)
        repo_df.to_csv(os.path.join(QUERY_CVE_DIR, f'{owner}_{repo}.csv'), index=False)
        print('save to', os.path.join(QUERY_CVE_DIR, f'{owner}_{repo}.csv'))
    
    


if __name__ == '__main__':
    
    # check_rows()
    
    # temp_check()
    # process_test_data(ranking_file='/mnt/local/Baselines_Bugs/ColBERT/retrieve_output_500000/commits_train/retrieve.py/2023-09-22_10.24.02/ranking.tsv')
    # pid_qid_test_data()
    # test_tsv_format()
    
    collect_top100_all()
    # split_data_by_repo()
