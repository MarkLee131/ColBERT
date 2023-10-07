import os
import pandas as pd

TOP100_DIR = '/mnt/local/Baselines_Bugs/ColBERT/data/cve_split/top100_data'
TOP100_SPLIT_DIR = '/mnt/local/Baselines_Bugs/ColBERT/data/cve_split/top100_split'
os.makedirs(TOP100_SPLIT_DIR, exist_ok=True)


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
    results_df.to_csv('/mnt/local/Baselines_Bugs/ColBERT/data/cve_split/results.csv', index=False)

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


def split_data():
    # qid,pid,rank,desc_token,commits,owner,repo,commit_id,cve,label
    train_top100 = pd.DataFrame(columns=['qid', 'pid', 'rank', 'desc_token', 'commits', 'owner', 'repo', 'commit_id', 'cve', 'label'])
    validate_top100 = pd.DataFrame(columns=['qid', 'pid', 'rank', 'desc_token', 'commits', 'owner', 'repo', 'commit_id', 'cve', 'label'])
    test_top100 = pd.DataFrame(columns=['qid', 'pid', 'rank', 'desc_token', 'commits', 'owner', 'repo', 'commit_id', 'cve', 'label'])
    
    test_df = pd.read_csv('/mnt/local/Baselines_Bugs/ColBERT/data/test_data.csv')
    validate_df = pd.read_csv('/mnt/local/Baselines_Bugs/ColBERT/data/validate_data.csv')
    # train_df = pd.read_csv('/mnt/local/Baselines_Bugs/ColBERT/data/train_data.csv')
    validate_cves = validate_df['cve'].unique()
    validate_tuples = validate_df[['cve', 'owner', 'repo']].drop_duplicates()
    
    test_cves = test_df['cve'].unique()
    test_tuples = test_df[['cve', 'owner', 'repo']].drop_duplicates()
    
    print('Number of validate CVEs: ', len(validate_cves))
    print('Number of test CVEs: ', len(test_cves))
    
    for file in os.listdir(TOP100_DIR):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(TOP100_DIR, file))
            ### check the cve in df, and determine which set it belongs to
            df_cves = df.groupby(['cve', 'owner', 'repo'])
            for (cve, owner, repo), group in df_cves:
                # if cve in validate_cves:
                if (cve, owner, repo) in validate_tuples.values:
                    validate_top100 = validate_top100.append(group, ignore_index=True)
                # elif cve in test_cves:
                elif (cve, owner, repo) in test_tuples.values:
                    test_top100 = test_top100.append(group, ignore_index=True)
                else:
                    train_top100 = train_top100.append(group, ignore_index=True)

    ### count the cves in each set
    print('Number of validate CVEs: ', validate_top100['cve'].nunique())
    print('Number of test CVEs: ', test_top100['cve'].nunique())
    print('Number of train CVEs: ', train_top100['cve'].nunique())


    ### save to csv files
    train_top100.to_csv(os.path.join(TOP100_SPLIT_DIR, 'train_top100.csv'), index=False)
    validate_top100.to_csv(os.path.join(TOP100_SPLIT_DIR, 'validate_top100.csv'), index=False)
    test_top100.to_csv(os.path.join(TOP100_SPLIT_DIR, 'test_top100.csv'), index=False)
      

#### calculate the metrics for the test data      
def compute_metrics(df, k_values):
    recalls = {k: [] for k in k_values}
    mrrs = []
    manual_efforts = {k: [] for k in k_values}

    # grouped = df.groupby('qid')
    # print(len(grouped))
    grouped = df.groupby(['cve', 'owner', 'repo'])
    print(len(grouped))

    for (cve, owner, repo), group in grouped:
        group_sorted = group.sort_values(by='rank')
        positive_ranks = group_sorted[group_sorted['label'] == 1]['rank'].tolist()

        for k in k_values:
            top_k_counts = sum(1 for rank in positive_ranks if rank <= k)
            recalls[k].append(top_k_counts / len(positive_ranks) if positive_ranks else 0)

            if positive_ranks:
                min_rank_within_k = min((rank for rank in positive_ranks if rank <= k), default=k)
            else:
                min_rank_within_k = k
            manual_efforts[k].append(min_rank_within_k)

        reciprocal_ranks = [1 / rank for rank in positive_ranks]
        mrrs.append(sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0)

    avg_recalls = {k: sum(recalls[k]) / len(recalls[k]) if recalls[k] else 0 for k in k_values}
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0
    avg_manual_efforts = {k: sum(manual_efforts[k]) / len(manual_efforts[k]) if manual_efforts[k] else 0 for k in k_values}

    return avg_recalls, avg_mrr, avg_manual_efforts

def save_metrics_to_csv(avg_recalls, avg_mrr, avg_manual_efforts, save_path):
    data = {
        'k': list(avg_recalls.keys()),
        'recall': [avg_recalls[k] for k in avg_recalls],
        'manual_effort': [avg_manual_efforts[k] for k in avg_recalls],
        'MRR': [avg_mrr for _ in avg_recalls]
    }
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)

def evaluate_test_data(data_path, output_path):
    df = pd.read_csv(data_path)
    k_values = [1,2,3,4,5,6,7,8,9,10,20,30,50,100]
    avg_recalls, avg_mrr, avg_manual_efforts = compute_metrics(df, k_values)
    save_metrics_to_csv(avg_recalls, avg_mrr, avg_manual_efforts, output_path)
    return avg_recalls, avg_mrr, avg_manual_efforts

if __name__ == '__main__':
    
    # calculate_metrics()
    
    split_data()
    # # Usage:
    data_path = "/mnt/local/Baselines_Bugs/ColBERT/data/cve_split/top100_split/test_top100.csv"
    # data_path = "/mnt/local/Baselines_Bugs/PatchSleuth/metrics/CR_1004/rank_info_final_model.csv"
    output_path = "testdata_results_colbert_1007.csv"
    results = evaluate_test_data(data_path, output_path)
    print(results)
    
    