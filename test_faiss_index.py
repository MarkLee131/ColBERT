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
subprocess.run(["python", "-m", "colbert.index_faiss",
                "--index_root", "/mnt/local/Baselines_Bugs/ColBERT/commits_indexes",
                "--index_name", "train_index",
                "--partitions", "4715",
                "--sample", "0.3",
                "--root", "index_output",
                "--experiment", "commits_train"])



