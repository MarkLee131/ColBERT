'''
This script is used to process the error data when indexing and faiss search.
The primary idea is parse the error message and find the corresponding commit repo, and index again.
'''
import os
import re

log_path = '/mnt/local/Baselines_Bugs/ColBERT/data/cve_split_scripts/run_colbert.log'
BASE_DIR = '/mnt/local/Baselines_Bugs/ColBERT/data/cve_split/collection_data'
RETRIEVAL_DIR = '/mnt/local/Baselines_Bugs/ColBERT/cve/retrieve_output/commits_train'

def parse_log(log_path=log_path):
    f = open(log_path, 'r')
    ## find the error line  `RuntimeError: Error in faiss::FileIOReader::FileIOReader(const char*) at /home/conda/feedstock_root/build_artifacts/faiss-split_1618468141526/work/faiss/impl/io.cpp:82: Error: 'f' failed: could not open /mnt/local/Baselines_Bugs/ColBERT/run_index/AcademySoftwareFoundation_openexr/AcademySoftwareFoundation_openexr/ivfpq.70.faiss for reading: No such file or directory`
    
    res = []
    for line in f.readlines():
        if 'RuntimeError' in line:
            repo = re.findall(r'run_index/(.*?)/', line)[0]
            res.append(repo)
    print(res)
    print(len(res))
    
    
index_dir = '/mnt/local/Baselines_Bugs/ColBERT/run_index'

def search_index(index_dir=index_dir):
    '''
    index_dir: the directory of index files
    we need to check whether the each subfolder has index files (*.pt for index and *.fiass for faiss index)
    '''
    index_res = []
    faiss_res = []
    for root, dirs, _ in os.walk(index_dir):
        for dir in dirs:
        ### check the index files in each subfolder
            # print(os.path.join(root, dir))
            if not os.path.exists(os.path.join(root, dir)):
                print(os.path.join(root, dir))
                index_res.append(dir)
                print('no such dir:', os.path.join(root, dir))
                continue
            
            ### we need to search the index files in the subfolder root/dir/dir
            # print(os.path.join(root, dir))
            pt_files = [file for _, _, files in os.walk(os.path.join(root, dir)) for file in files if file.endswith('.pt')]
            faiss_files = [file for _, _, files in os.walk(os.path.join(root, dir)) for file in files if file.endswith('.faiss')]
            if len(pt_files) == 0:
                # index_res.append(os.path.join(root, dir))
                index_res.append(dir)

            if len(faiss_files) == 0:
                faiss_res.append(os.path.join(root, dir))

            
    print('index_res:', index_res)
    print('len(index_res):', len(index_res))
    # print('faiss_res:', faiss_res)
    print('len(faiss_res):', len(faiss_res))
    
    # left_faiss = set(faiss_res)-set(index_res)
    
    # list_left_faiss = [index.split('/')[-1] for index in left_faiss]
    # print('list_left_faiss:', list_left_faiss)
                
import os

def has_ranking(root, dir):
    '''
    Check whether a `ranking.tsv` file exists in any subfolder of root/dir/
    '''
    target_path = os.path.join(root, dir)
    
    for foldername, subfolders, filenames in os.walk(target_path):
        if 'ranking.tsv' in filenames:
            return True

    return False


def check_retrieval(retriveal_dir='/mnt/local/Baselines_Bugs/ColBERT/run/retrieve_output'):
    missing_retrieval = []
    
    # Get top-level directories under retriveal_dir
    top_level_dirs = [d for d in os.listdir(retriveal_dir) if os.path.isdir(os.path.join(retriveal_dir, d))]
    
    for dir in top_level_dirs:
        if dir == 'logs':
            continue
        # Check for ranking.tsv in the subfolder root/dir
        if not has_ranking(retriveal_dir, dir):
            missing_retrieval.append(dir)

    print('missing_retrieval:', missing_retrieval)
    print('len(missing_retrieval):', len(missing_retrieval))

import pandas as pd
def count_line_number(file_path):
    '''
    Count the number of lines in a tsv file
    '''
    df = pd.read_csv(file_path, sep='\t', header=None)
    df.columns = ['qid', 'pid', 'rank']
    df_group = df.groupby('qid').count()
    return df_group['pid'].values

def search_ranking_tsv():
    '''
    Search for ranking.tsv in the subfolder root/dir, and append the path to the file to the list
    '''
    line_info = {}

    top_level_dir = [d for d in os.listdir(RETRIEVAL_DIR) if os.path.isdir(os.path.join(RETRIEVAL_DIR, d))]

    cnt = 0
    total_num = 0
    cve_cnt = 0
    for dir in top_level_dir:
        if dir == 'logs':
            continue
        for foldername, subfolders, filenames in os.walk(os.path.join(RETRIEVAL_DIR, dir)):
            if 'ranking.tsv' in filenames:
                file_path = os.path.join(foldername, 'ranking.tsv')
                line_number = count_line_number(file_path)
                total_num += sum(list(line_number))
                line_info[dir] = line_number
                cve_cnt += len(list(line_number))
                cnt += 1
                print(cnt)

    # save line_info to a csv file
    df = pd.DataFrame(list(line_info.items()), columns=['repo', 'line_number'])
    df.to_csv('line_info.csv', index=False)
    print('save line_info to ./line_info.csv')
    print('total_num:', total_num)
    print('cve_cnt:', cve_cnt)
    
def copy_rankingfile():
    '''
    Copy the ranking.tsv file to the corresponding repo folder
    '''
    top_level_dir = [d for d in os.listdir(RETRIEVAL_DIR) if os.path.isdir(os.path.join(RETRIEVAL_DIR, d))]

    ranking_dir = '/mnt/local/Baselines_Bugs/ColBERT/data/ranking_data'  
    cnt = 0
    for dir in top_level_dir:
        if dir == 'logs':
            continue
        for foldername, subfolders, filenames in os.walk(os.path.join(RETRIEVAL_DIR, dir)):
            if 'ranking.tsv' in filenames:
                file_path = os.path.join(foldername, 'ranking.tsv')
                os.system('cp {} {}'.format(file_path, os.path.join(ranking_dir, dir+'.tsv')))
                cnt += 1
                print(cnt)
                break
    print('copy {} ranking.tsv files'.format(cnt))
    

                
        
    
    
    
    

if __name__ == '__main__':
    # parse_log()
    
    # search_index()
    
    # check_retrieval()
    
    # search_ranking_tsv()
    # copy_rankingfile()