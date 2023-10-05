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
                "--amp", "--doc_maxlen", "512", "--mask-punctuation", "--bsize", "128",
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
                "--sample", "0.3",
                "--partitions", "70",
                # "--similarity", "l2",
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
                        "--root", "cve/retrieve_output", "--experiment", "commits_train"]

    retrieval_res = subprocess.run(retrieval_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    # print(retrieval_res.stdout)
    logging.info(retrieval_res.stdout)
    
    if retrieval_res.stderr:
        print(retrieval_res.stderr)
        logging.info(retrieval_res.stderr)
        # print("retrieval failed")
        # return False

    
    
        
if __name__ == "__main__":

    ### run the index 
    collection_files = []
    for collection_file in os.listdir(COLLECTION_DIR):
        if collection_file.endswith(".tsv"):
            collection_files.append(os.path.join(COLLECTION_DIR, collection_file))
    
    for collection_file in tqdm(collection_files, desc="indexing", total=len(collection_files)):
        index_name = collection_file.split("/")[-1].replace(".tsv", "")
        index_root = os.path.join("/mnt/local/Baselines_Bugs/ColBERT/cve_index", index_name)
        
        print("indexing {}".format(index_name))
        logging.info("indexing {}".format(index_name))
        run_index(collection_file, index_root, index_name)
        
        print("indexing faiss {}".format(index_name))
        logging.info("indexing faiss {}".format(index_name))
        run_faiss_index(index_root, index_name)
        
        print("retrieving {}".format(index_name))
        logging.info("retrieving {}".format(index_name))
        queries_file = os.path.join(QUERY_DIR, index_name + ".tsv")
        run_retrieval(queries_file, index_root, index_name, "100")

        
    