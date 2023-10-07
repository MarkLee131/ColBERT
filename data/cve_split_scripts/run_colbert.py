import os
import subprocess
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', filename='/mnt/local/Baselines_Bugs/ColBERT/data/cve_split_scripts/run_colbert.log', filemode='w')
COLLECTION_DIR = "/mnt/local/Baselines_Bugs/ColBERT/data/cve_split/collection_data"
QUERY_DIR = "/mnt/local/Baselines_Bugs/ColBERT/data/cve_split/query_data"

### set environment variables
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["OMP_NUM_THREADS"] = "6"

def run_index(collection_file, index_root, index_name):

    index_cmd = ["python", "-m", "torch.distributed.launch", "--nproc_per_node=4", "-m", "colbert.index",
                "--amp", 
                # "--query_maxlen", "128", # default 32
                "--doc_maxlen", "512", "--mask-punctuation", "--bsize", "128",
                "--checkpoint", "/mnt/local/Baselines_Bugs/ColBERT/commits_exp/commits_train/train.py/test.l2/checkpoints/colbert.dnn",
                "--collection", collection_file,
                "--similarity", "l2",
                "--index_root", index_root, "--index_name", index_name,
                "--root", "cve/index_output", "--experiment", "commits_train"]
    # print("index_cmd: {}".format(index_cmd))

    index_res = subprocess.run(index_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    print(index_res.stdout)
    
    if index_res.stderr:
        print(index_res.stderr)
        logging.info(index_res.stderr)
        # print("indexing failed")
    
    print("indexing done")
    

def run_faiss_index(index_root, index_name):
    index_faiss_cmd = ["python", "-m", "colbert.index_faiss",
                "--index_root", index_root, "--index_name", index_name,
                # "--sample", "0.3",
                "--partitions", "70",
                # "--similarity", "l2",
                "--sample", "0.8",
                "--root", "cve/faiss_output",
                "--experiment", "commits_train"]
    index_res = subprocess.run(index_faiss_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    print(index_res.stdout)
    logging.info(index_res.stdout)
    
    if index_res.stderr:
        print(index_res.stderr)
        logging.info(index_res.stderr)
        # print("indexing faiss failed")
    print("indexing faiss done")




def run_retrieval(queries_file, index_root, index_name, topk):
    retrieval_cmd = ["python", "-m", "colbert.retrieve",
                        "--amp", "--doc_maxlen", "512", "--mask-punctuation", "--bsize", "128",
                        "--queries", queries_file,
                        "--nprobe", "32", 
                        "--partitions", "70", 
                        # "--faiss_depth", "100", 
                        "--depth", topk, 
                        "--similarity", "l2",
                        "--index_root", index_root, "--index_name", index_name,
                        "--checkpoint", "/mnt/local/Baselines_Bugs/ColBERT/commits_exp/commits_train/train.py/test.l2/checkpoints/colbert.dnn",
                        "--root", "cve/retrieve_output",
                        # "--experiment", index_name
                        "--experiment", "commits_train"
                        ]
    ### !!!caution!!!: experiment name should be the same as index_name, otherwise the retrieval result will be saved into a folder named timestamp.
    retrieval_res = subprocess.run(retrieval_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    # print(retrieval_res.stdout)
    logging.info(retrieval_res.stdout)
    
    if retrieval_res.stderr:
        print(retrieval_res.stderr)
        logging.info(retrieval_res.stderr)
        # print("retrieval failed")
        # return False

###
def check_indexed(index_root, index_name):
    index_path = os.path.join(index_root, index_name)
    # check whether the index files exist under the index_path(is a dir)   
    if os.path.exists(index_path) and len(os.listdir(index_path)) > 0 and os.path.exists(os.path.join(index_path, "ivfpq.70.faiss")):
        ## index files exist and faiss indexed    
        return True, True
    
    elif os.path.exists(index_path) and len(os.listdir(index_path)) > 0 and os.path.exists(os.path.join(index_path, "0.pt")):
        ## index files exist but not faiss indexed    
        return True, False
    else:
        ## index files not exist
        return False, False
    
def check_retriveal(
                    retrieval_root: str="/mnt/local/Baselines_Bugs/ColBERT/cve/retrieve_output/commits_train/retrieve.py"
                    ):
    '''
    check whether the retrieval files exist under the retrieval_path, 
    and collect the ranking.tsv by copying them into a new folder rename as index_name
    '''
    cnt = 0
    missing_cnt = 0
    for folder in os.listdir(retrieval_root):
        ranking_path = os.path.join(retrieval_root, folder, "ranking.tsv")
        cnt += 1
        if not os.path.exists(ranking_path):
            # print("{} not exist".format(ranking_path))
            folder_path = os.path.join(retrieval_root, folder)
            # logging.info("ranking.tsv not exist in {}".format(folder_path))
            print("ranking.tsv not exist in {}".format(folder_path))
            missing_cnt += 1
    print("total: {}, missing: {}".format(cnt, missing_cnt))

import json
def move_rankings(source_dir:str="/mnt/local/Baselines_Bugs/ColBERT/cve/retrieve_output/commits_train/retrieve.py",
                  save_dir:str="/mnt/local/Baselines_Bugs/ColBERT/data/cve_split/ranking_data"
                    ):
    query_data_dir = "/mnt/local/Baselines_Bugs/ColBERT/data/cve_split/query_data"
    repos = []
    os.makedirs(save_dir, exist_ok=True)
    for file in os.listdir(query_data_dir):
        if file.endswith(".tsv"):
            index_name = file.replace(".tsv", "")
            repos.append(index_name)
    print("num of repos: {}".format(len(repos)))
    
    for folder in os.listdir(source_dir):
        ranking_path = os.path.join(source_dir, folder, "ranking.tsv")
        if os.path.exists(ranking_path):
            with open(os.path.join(source_dir, folder, "logs", "args.json"), "r") as f:
                args = json.load(f)
                repo = args["index_name"]
            index_name = repo
            if index_name in repos:
                save_path = os.path.join(save_dir, index_name + ".tsv")
                # copy ranking.tsv to save_path
                os.system("cp {} {}".format(ranking_path, save_path))
                repos.remove(index_name)
            else:
                print("index_name {} not in repos".format(index_name))
                
    print("num of repos not found: {}".format(len(repos)))
    print(repos)
            
            
            

if __name__ == "__main__":

    # # ## run the index 
    # # collection_files = []
    # # for collection_file in os.listdir(COLLECTION_DIR):
    # #     if collection_file.endswith(".tsv"):
    # #         collection_files.append(os.path.join(COLLECTION_DIR, collection_file))
    
    # missing_files = ['01org_tpm2.0-tools', 'varnish_Varnish-Cache']
    # # missing_files = [ 'lighttpd_lighttpd1.4.tsv', 'oetiker_rrdtool-1.x.tsv'] 
    # collection_files = [os.path.join(COLLECTION_DIR, file) for file in missing_files]
    
    # for collection_file in tqdm(collection_files, desc="indexing", total=len(collection_files)):
    #     index_name = collection_file.split("/")[-1].replace(".tsv", "")
    #     index_root = os.path.join("/mnt/local/Baselines_Bugs/ColBERT/cve_index", index_name)
        
    #     # print("indexing {}".format(index_name))
    #     # logging.info("indexing {}".format(index_name))
    #     # run_index(collection_file, index_root, index_name)
        
    #     # print("indexing faiss {}".format(index_name))
    #     # logging.info("indexing faiss {}".format(index_name))
    #     # run_faiss_index(index_root, index_name)
        
    #     print("retrieving {}".format(index_name))
    #     logging.info("retrieving {}".format(index_name))
    #     queries_file = os.path.join(QUERY_DIR, index_name + ".tsv")
    #     run_retrieval(queries_file, index_root, index_name, "100")

        
    # ###########################################################################
    # ## test whether all the repositories are indexed and retrieved successfully
    # check_retriveal()
    
    move_rankings()