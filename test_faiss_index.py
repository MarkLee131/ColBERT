### we need to use this script to replace the below command:
'''
```bash
#### set the rank, we now use one machine with 4 GPUs
export RANK=0
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

python -m colbert.index_faiss \
--index_root commits_indexes --index_name train_index \
--partitions 4715 --sample 0.3 \
--root index_output --experiment commits_train
```
'''

import os

### set environment variables
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

### import the script
'''
python -m colbert.index_faiss \
--index_root commits_indexes --index_name train_index \
--partitions 4715 --sample 0.3 \
--root index_output --experiment commits_train
'''
import subprocess


#### faiss index
# subprocess.run(["python", "-m", "colbert.index_faiss",
#                 "--index_root", "/mnt/local/Baselines_Bugs/ColBERT/commits_indexes",
#                 "--index_name", "train_index",
#                 "--partitions", "4715",
#                 "--sample", "0.3",
#                 "--root", "index_output",
#                 "--experiment", "commits_train"])


'''
python -m colbert.retrieve \
--amp --doc_maxlen 512 --query_maxlen 512 --mask-punctuation --bsize 256 \
--queries /mnt/local/Baselines_Bugs/ColBERT/data/queries_all.tsv \
--nprobe 32 --partitions 4715 --faiss_depth 100 \
--index_root /mnt/local/Baselines_Bugs/ColBERT/commits_indexes --index_name train_index \
--checkpoint /mnt/local/Baselines_Bugs/ColBERT/commits_exp/commits_train/train.py/test.l2/checkpoints/colbert.dnn \
--root retrieve_output --experiment commits_train
'''

def run_retrive(topk=10000):
    subprocess.run(["python", "-m", "colbert.retrieve",
                "--amp", "--doc_maxlen", "512", "--mask-punctuation", "--bsize", "1024",
                "--queries", "/mnt/local/Baselines_Bugs/ColBERT/data/queries_all.tsv",
                "--nprobe", "32", "--partitions", "4715", "--faiss_depth", str(topk),
                "--depth", str(topk), 
                "--index_root", "/mnt/local/Baselines_Bugs/ColBERT/commits_indexes", "--index_name", "train_index",
                "--checkpoint", "/mnt/local/Baselines_Bugs/ColBERT/commits_exp/commits_train/train.py/test.l2/checkpoints/colbert.dnn",
                "--root", "retrieve_output_" + str(topk), "--experiment", "commits_train"])

if __name__=='__main__':

    # run_retrive(topk=5000)
    
    # run_retrive(topk=50000)
    
    run_retrive(topk=500000)
    
    # run_retrive(topk=1000000)