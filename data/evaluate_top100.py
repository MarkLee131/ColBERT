import os
import pandas as pd

TOP100_DIR = '/mnt/local/Baselines_Bugs/ColBERT/data/top100'

def calculate_metrics():
    '''
    Calculate the Recall, MRR, and Manual efforts
    '''
    
    ranks = []
    cves = 0
    queries = 0
    
    for file in os.listdir(TOP100_DIR):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(TOP100_DIR, file))
            cves += df['cve'].nunique()
            queries += df['qid'].nunique()
            patch_df = df[df['label']==1]
            ### we need to collect the ranks of the patches
            
            for qid in df['qid'].unique():
                if qid in patch_df['qid'].unique():
                    ### we need the min rank of the patches for a CVE
                    ranks.append(min(patch_df[patch_df['qid']==qid]['rank']))
                else:
                    ranks.append(0)

                        
    print('Number of CVEs: ', cves)
    print('Number of queries: ', queries)
    print('Number of patches: ', len(ranks))
    # mrr = sum([1/rank for rank in ranks])/len(ranks)
    mrr = 0
    for rank in ranks:
        if rank != 0:
            mrr += 1/rank
        if rank == 0:
            mrr += 1/100
            
    mrr = mrr/len(ranks)
    print('MRR: ', mrr)
    
    for topk in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100]:
    # for topk in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 60, 70, 80, 90, 99, 100]:
        print('Recall@{}: {}'.format(topk, recall_k(ranks, topk)))
        print('Manual Efforts@{}: {}'.format(topk, manual_efforts_k(ranks, topk)))

    ### save the printable results into a csv file
    
    results_df = pd.DataFrame(columns=['k', 'recall@k', 'manual efforts@k', 'mrr'])
    # for topk in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 60, 70, 80, 90, 99, 100]:
    for topk in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100]:
        results_df = results_df.append({'k': topk, 'recall@k': recall_k(ranks, topk), 'manual efforts@k': manual_efforts_k(ranks, topk), 'mrr': mrr}, ignore_index=True)
    results_df.to_csv('/mnt/local/Baselines_Bugs/ColBERT/data/results.csv', index=False)

def recall_k(ranks, k):
    '''
    Calculate the recall@k
    '''
    TP = 0
    for rank in ranks:
        if rank <= k and rank != 0:
            TP += 1
    return TP/len(ranks)

def manual_efforts_k(ranks, k):
    '''
    Calculate the manual effort@k
    '''
    manual_efforts_k = 0
    for rank in ranks:
        if rank <= k and rank != 0:
            manual_efforts_k += rank
        else:
            manual_efforts_k += k
        
    return manual_efforts_k/len(ranks)


if __name__ == '__main__':
    calculate_metrics()