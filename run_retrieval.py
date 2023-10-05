import gc
import os
import subprocess
import pandas as pd
from data.data_prepare import reduce_mem_usage
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', filename='run_retrieval.log', filemode='w')
COLLECTION_DIR = "/mnt/local/Baselines_Bugs/ColBERT/data/repo_data"
QUERY_DIR = "/mnt/local/Baselines_Bugs/ColBERT/data/query_data"

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
                "--root", "run/index_output", "--experiment", "commits_train"]
    # print("index_cmd: {}".format(index_cmd))

    # index_faiss_cmd = ["python", "-m", "colbert.index_faiss",
    #             "--index_root", index_root, "--index_name", index_name,
    #             "--sample", "0.3",
    #             "--partitions", "70",
    #             "--root", "run/faiss_output",
    #             "--experiment", "commits_train"]
    index_res = subprocess.run(index_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    print(index_res.stdout)
    
    if index_res.stderr:
        print(index_res.stderr)
        logging.info(index_res.stderr)
        # print("indexing failed")
    #     return 
    # print("indexing done")
    
    # print("start indexing faiss")
    # index_res = subprocess.run(index_faiss_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    # print(index_res.stdout)
    
    # if index_res.stderr:
    #     print(index_res.stderr)
    #     print("indexing faiss failed")
    # print("indexing faiss done")

def run_faiss_index(index_root, index_name):
    index_faiss_cmd = ["python", "-m", "colbert.index_faiss",
                "--index_root", index_root, "--index_name", index_name,
                "--sample", "0.3",
                "--partitions", "70",
                "--similarity", "l2",
                "--root", "run/faiss_output",
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
                        "--root", "run/retrieve_output", "--experiment", "commits_train"]

    retrieval_res = subprocess.run(retrieval_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    # print(retrieval_res.stdout)
    logging.info(retrieval_res.stdout)
    
    if retrieval_res.stderr:
        print(retrieval_res.stderr)
        logging.info(retrieval_res.stderr)
        # print("retrieval failed")
        # return
        
        
'''
python -m colbert.rerank \
--root rerank_output --experiment commits_train \
--amp --bsize 256 \
--query_maxlen 512 --doc_maxlen 512 --mask-punctuation \
--checkpoint /mnt/local/Baselines_Bugs/ColBERT/commits_exp/commits_train/train.py/test.l2/checkpoints/colbert.dnn \
--topk 100 \
--index_root /mnt/local/Baselines_Bugs/ColBERT/commits_indexes --index_name train_index \
--queries /mnt/local/Baselines_Bugs/ColBERT/data/queries_all.tsv \
--collection /mnt/local/Baselines_Bugs/ColBERT/data/collection_all.tsv
'''

def run_rerank(queries_file, index_root, index_name, topk, rerank_rankname):
    rerank_cmd = ["python", "-m", "colbert.rerank",
                        "--amp", "--doc_maxlen", "512", "--mask-punctuation", "--bsize", "128",
                        "--queries", queries_file,
                        # "--nprobe", "32", 
                        # "--partitions", "70", 
                        # "--faiss_depth", "100", 
                        "--topk", topk, 
                        "--index_root", index_root, "--index_name", index_name,
                        "--checkpoint", "/mnt/local/Baselines_Bugs/ColBERT/commits_exp/commits_train/train.py/test.l2/checkpoints/colbert.dnn",
                        "--root", "run/rerank_output", "--experiment", index_name]

    rerank_res = subprocess.run(rerank_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    print(rerank_res.stdout)
    
    if rerank_res.stderr:
        print(rerank_res.stderr)
        print("rerank failed")
        # return 
    print("rerank done")
    
    

def split_repo():
    DATA_DIR = "/mnt/local/Baselines_Bugs/ColBERT/data"
    SAVE_DIR = "/mnt/local/Baselines_Bugs/ColBERT/data/repo_data"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    train_file = os.path.join(DATA_DIR, "train_data.csv")
    validate_file = os.path.join(DATA_DIR, "validate_data.csv")
    test_file = os.path.join(DATA_DIR, "test_data.csv")
    
    
    test_df = pd.read_csv(test_file)
    reduce_mem_usage(test_df)
    
    validate_df = pd.read_csv(validate_file)
    reduce_mem_usage(validate_df)
    
    train_df = pd.read_csv(train_file)
    reduce_mem_usage(train_df)
    
    ## concate them and split them by their 'owner' and 'repo'
    all_df = pd.concat([train_df, test_df], ignore_index=True, axis=0)
    del train_df, test_df
    
    all_df = pd.concat([validate_df, all_df], ignore_index=True, axis=0)
    del validate_df
    reduce_mem_usage(all_df)
    gc.collect()

    
    repo_df = all_df.groupby(['owner', 'repo'])
    ## count the number of repos
    print("number of repos: {}".format(len(repo_df)))

    ## save the rows of each repo into a single file
    for (owner, repo), group in tqdm(repo_df, desc="splitting repos"):
        # print(owner, repo)
        # print(group.shape)
        # print(group.columns)
        save_file = os.path.join(SAVE_DIR, owner + "_" + repo + ".csv")
        # adjust the column order and set the index to be the 'pid'
        group['pid'] = range(0, len(group))
        # replace the \t by using ' '
        group['desc_token'] = group['desc_token'].str.replace("\t", " ", regex=False)
        group['commits'] = group['msg_token'].str.cat(group['diff_token'], sep=" ").str.replace("\t", " ", regex=False)        
        group = group[['pid', 'commits', 'owner', 'repo', 'commit_id', 'cve', 'label']]
        
        group.to_csv(save_file, index=False)
        print("save to {}".format(save_file))


def process2tsv():
    files = []
    for f in os.listdir("/mnt/local/Baselines_Bugs/ColBERT/data/repo_data"):
        if f.endswith(".csv"):
            files.append(os.path.join("/mnt/local/Baselines_Bugs/ColBERT/data/repo_data", f))
    print("number of files: {}".format(len(files)))
    
    for f in tqdm(files, desc="processing files"):
        df = pd.read_csv(f)
        
        tmp_df = df.head(4999) ### if need to rerun, we need this line to limit the number of rows
        patch_df = df[df['label'] == 1]
        df = pd.concat([tmp_df, patch_df], ignore_index=True, axis=0)
        
        # print("processing {}".format(f))
        # remove the header
        df.to_csv(f.replace(".csv", ".tsv"), sep="\t", index=False, header=False)
        print("save to {}".format(f.replace(".csv", ".tsv")))
        # break
        
def split_queries():
    DATA_DIR = "/mnt/local/Baselines_Bugs/ColBERT/data"
    # SAVE_DIR = "/mnt/local/Baselines_Bugs/ColBERT/data/repo_data"
    # SAVE_DIR = "/mnt/local/Baselines_Bugs/ColBERT/data/query_data"
    SAVE_DIR = "/mnt/local/Baselines_Bugs/ColBERT/data/query_data_extra"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    train_file = os.path.join(DATA_DIR, "train_data.csv")
    validate_file = os.path.join(DATA_DIR, "validate_data.csv")
    test_file = os.path.join(DATA_DIR, "test_data.csv")
    
    # quary_df = pd.read_csv(, chunksize=5001)
    queries_dict = {}
    
    test_df = pd.read_csv(test_file, chunksize=5001)
    # reduce_mem_usage(test_df)
    train_df = pd.read_csv(train_file, chunksize=5001)
    validate_df = pd.read_csv(validate_file, chunksize=5001)
    
    for df in [test_df, validate_df, train_df]:
        for chunk in tqdm(df, desc="processing test"):
            owner_repo_group = chunk.groupby(['owner', 'repo'])
            for (owner, repo), group in owner_repo_group:
                # print(owner, repo)
                # print(group.shape)
                # print(group.columns)
                key = f'{owner}_{repo}'
                if key not in queries_dict:
                    queries_dict[key] = []
                queries_dict[key].extend(list(set(group['desc_token'].tolist())))

    for key, queries in tqdm(queries_dict.items(), desc="saving test"):
        df_temp = pd.DataFrame({"queries": list(set(queries))})
        df_temp['qid'] = range(0, len(df_temp))
        df_temp = df_temp[['qid', 'queries']]
        df_temp['queries'] = df_temp['queries'].str.replace("\t", " ", regex=False)
        df_temp.to_csv(os.path.join(SAVE_DIR, key + ".tsv"), sep="\t", index=False, header=False)
        



    
    
        
if __name__ == "__main__":
    ### split the data into different files by repo, but in csv format
    # split_repo()
    
    ## process the data into tsv format
    # process2tsv()
    
    
    # split_queries()
    
    
    # ### run the index 
    # collection_files = []
    # for collection_file in os.listdir(COLLECTION_DIR):
    #     if collection_file.endswith(".tsv"):
    #         collection_files.append(os.path.join(COLLECTION_DIR, collection_file))
    
    # for collection_file in tqdm(collection_files, desc="indexing", total=len(collection_files)):
    #     index_name = collection_file.split("/")[-1].replace(".tsv", "")
    #     index_root = os.path.join("/mnt/local/Baselines_Bugs/ColBERT/run_index", index_name)
        
    #     # print("indexing {}".format(index_name))
    #     # run_index(collection_file, index_root, index_name)
        
    #     print("retrieving {}".format(index_name))
    #     logging.info("retrieving {}".format(index_name))
    #     queries_file = os.path.join(QUERY_DIR, index_name + ".tsv")
    #     run_retrieval(queries_file, index_root, index_name, "100", "commits_train")
    #     # break
    

    # ### test rerank by using a repo
    # index_name = "01org_opa-fm"
    # index_root = os.path.join("/mnt/local/Baselines_Bugs/ColBERT/run_index", index_name)
    # queries_file = os.path.join(QUERY_DIR, index_name + ".tsv")
    # run_rerank(queries_file, index_root, index_name, "100", "commits_train")
    
    
    ##### 2023.10.01
    # # process the missing 6 repos when faiss indexing
    
    # index_list = ['varnish_Varnish-Cache', 'joyent_node', 'Uninett_mod_auth_mellon', 'varnish_Varnish-Cache', 'Uninett_mod_auth_mellon', 'joyent_node']
    
    # for index_name in index_list:
    #     index_root = os.path.join("/mnt/local/Baselines_Bugs/ColBERT/run_index", index_name)
    #     run_faiss_index(index_root, index_name)
        
        
    #### 2023.10.01
    ### RERUN the missing repos for indexing
    collection_files = []
    # missing_repos = ['AcademySoftwareFoundation_openexr', 'DanBloomberg_leptonica', 'Ettercap_ettercap', 'AntonKueltz_fastecdsa', 'FransUrbo_zfs', 'FreeRADIUS_freeradius-server', 'LibreDWG_libredwg', 'OSGeo_gdal', 'OpenRC_openrc', 'OpenSC_OpenSC', 'PCRE2Project_pcre2', 'SELinuxProject_selinux', 'Yubico_libu2f-host', 'bitlbee_bitlbee', 'bratsche_pango', 'brianmario_yajl-ruby', 'ccxvii_mujs', 'coturn_coturn', 'crawl_crawl', 'davidben_nspluginwrapper', 'dbry_WavPack', 'dlitz_pycrypto', 'facebook_folly', 'facebook_proxygen', 'facebook_wangle', 'facebookincubator_mvfst', 'fatcerberus_minisphere', 'glennrp_libpng', 'gsliepen_tinc', 'gssapi_gssproxy', 'h2o_h2o', 'hexchat_hexchat', 'keepkey_keepkey-firmware', 'kr_beanstalkd', 'laverdet_isolated-vm', 'libgit2_libgit2', 'libidn_libidn2', 'libofx_libofx', 'libtom_libtomcrypt', 'madler_pigz', 'maekitalo_tntnet', 'michaelrsweet_htmldoc', 'miniupnp_ngiflib', 'mjg59_linux', 'netblue30_firejail', 'netdata_netdata', 'ntop_nDPI', 'open5gs_open5gs', 'opencryptoki_opencryptoki', 'projectacrn_acrn-hypervisor', 'qpdf_qpdf', 'relan_exfat', 'rizinorg_rizin', 'shadowsocks_shadowsocks-libev', 'shellinabox_shellinabox', 'silnrsi_graphite', 'simsong_tcpflow', 'sroracle_abuild', 'stefanberger_swtpm', 'stephane_libmodbus', 'stoth68000_media-tree', 'strukturag_libde265', 'strukturag_libheif', 'swaywm_swaylock', 'swoole_swoole-src', 'symless_synergy-core', 'thorfdbg_libjpeg', 'torproject_tor', 'uWebSockets_uWebSockets', 'uclouvain_openjpeg', 'unrealircd_unrealircd', 'veracrypt_VeraCrypt', 'wesnoth_wesnoth', 'wireapp_wire-avs', 'wolfSSL_wolfMQTT', 'yast_yast-core', 'AcademySoftwareFoundation_openexr', 'DanBloomberg_leptonica', 'Ettercap_ettercap', 'AntonKueltz_fastecdsa', 'FransUrbo_zfs', 'FreeRADIUS_freeradius-server', 'LibreDWG_libredwg', 'OSGeo_gdal', 'OpenRC_openrc', 'OpenSC_OpenSC', 'PCRE2Project_pcre2', 'SELinuxProject_selinux', 'Yubico_libu2f-host', 'bitlbee_bitlbee', 'bratsche_pango', 'brianmario_yajl-ruby', 'ccxvii_mujs', 'coturn_coturn', 'crawl_crawl', 'davidben_nspluginwrapper', 'dbry_WavPack', 'dlitz_pycrypto', 'facebook_folly', 'facebook_proxygen', 'facebook_wangle', 'facebookincubator_mvfst', 'fatcerberus_minisphere', 'glennrp_libpng', 'gsliepen_tinc', 'gssapi_gssproxy', 'h2o_h2o', 'hexchat_hexchat', 'keepkey_keepkey-firmware', 'kr_beanstalkd', 'laverdet_isolated-vm', 'libgit2_libgit2', 'libidn_libidn2', 'libofx_libofx', 'libtom_libtomcrypt', 'madler_pigz', 'maekitalo_tntnet', 'michaelrsweet_htmldoc', 'miniupnp_ngiflib', 'mjg59_linux', 'netblue30_firejail', 'netdata_netdata', 'ntop_nDPI', 'open5gs_open5gs', 'opencryptoki_opencryptoki', 'projectacrn_acrn-hypervisor', 'qpdf_qpdf', 'relan_exfat', 'rizinorg_rizin', 'shadowsocks_shadowsocks-libev', 'shellinabox_shellinabox', 'silnrsi_graphite', 'simsong_tcpflow', 'sroracle_abuild', 'stefanberger_swtpm', 'stephane_libmodbus', 'stoth68000_media-tree', 'strukturag_libde265', 'strukturag_libheif', 'swaywm_swaylock', 'swoole_swoole-src', 'symless_synergy-core', 'thorfdbg_libjpeg', 'torproject_tor', 'uWebSockets_uWebSockets', 'uclouvain_openjpeg', 'unrealircd_unrealircd', 'veracrypt_VeraCrypt', 'wesnoth_wesnoth', 'wireapp_wire-avs', 'wolfSSL_wolfMQTT', 'yast_yast-core']
    # missing_repos = ['yast_yast-core'] ### testing
    
    ##### missing ranking.tsv
    missing_repos = ['AcademySoftwareFoundation_openexr', 'DanBloomberg_leptonica', 'Ettercap_ettercap', 'AntonKueltz_fastecdsa', 'FransUrbo_zfs', 'FreeRADIUS_freeradius-server', 'LibreDWG_libredwg', 'OSGeo_gdal', 'OpenRC_openrc', 'OpenSC_OpenSC', 'PCRE2Project_pcre2', 'SELinuxProject_selinux', 'Uninett_mod_auth_mellon', 'Yubico_libu2f-host', 'bitlbee_bitlbee', 'bratsche_pango', 'brianmario_yajl-ruby', 'ccxvii_mujs', 'coturn_coturn', 'crawl_crawl', 'davidben_nspluginwrapper', 'dbry_WavPack', 'dlitz_pycrypto', 'facebook_folly', 'facebook_proxygen', 'facebook_wangle', 'facebookincubator_mvfst', 'fatcerberus_minisphere', 'glennrp_libpng', 'gsliepen_tinc', 'gssapi_gssproxy', 'h2o_h2o', 'hexchat_hexchat', 'joyent_node', 'keepkey_keepkey-firmware', 'kr_beanstalkd', 'laverdet_isolated-vm', 'libgit2_libgit2', 'libidn_libidn2', 'libofx_libofx', 'libtom_libtomcrypt', 'madler_pigz', 'maekitalo_tntnet', 'michaelrsweet_htmldoc', 'miniupnp_ngiflib', 'mjg59_linux', 'netblue30_firejail', 'netdata_netdata', 'ntop_nDPI', 'open5gs_open5gs', 'opencryptoki_opencryptoki', 'projectacrn_acrn-hypervisor', 'qpdf_qpdf', 'relan_exfat', 'rizinorg_rizin', 'shadowsocks_shadowsocks-libev', 'shellinabox_shellinabox', 'silnrsi_graphite', 'simsong_tcpflow', 'sroracle_abuild', 'stefanberger_swtpm', 'stephane_libmodbus', 'stoth68000_media-tree', 'strukturag_libde265', 'strukturag_libheif', 'swaywm_swaylock', 'swoole_swoole-src', 'symless_synergy-core', 'thorfdbg_libjpeg', 'torproject_tor', 'uWebSockets_uWebSockets', 'uclouvain_openjpeg', 'unrealircd_unrealircd', 'varnish_Varnish-Cache', 'veracrypt_VeraCrypt', 'wesnoth_wesnoth', 'wireapp_wire-avs', 'wolfSSL_wolfMQTT', 'yast_yast-core']
    
    print("number of missing repos: {}".format(len(missing_repos)))
    
    collection_files = [os.path.join(COLLECTION_DIR, repo + ".tsv") for repo in missing_repos]
    
    for collection_file in tqdm(collection_files, desc="indexing or faiss indexing", total=len(collection_files)):
        index_name = collection_file.split("/")[-1].replace(".tsv", "")
        index_root = os.path.join("/mnt/local/Baselines_Bugs/ColBERT/run_index", index_name)
        
        # ### we need to first check whether the index files exist, i.e., index_root/index_name/ has files
        # if os.path.exists(os.path.join(index_root, index_name)) and len(os.listdir(os.path.join(index_root, index_name))) > 0:
        #     print("{} exists".format(index_root))
        #     logging.info("{} exists".format(index_root))
            
        #     continue
        # else:
        #     logging.info("remove {}".format(index_root))
        #     print("remove {}".format(index_root))
        #     ## remove it
        #     os.system("rm -rf {}".format(index_root))
        
        ###### remove directly, without checking
        # logging.info("indexing {}".format(index_name))
        # print("remove {}".format(index_root))
        # os.system("rm -rf {}".format(index_root))
        #######################################################
        
        # print("re-indexing {}".format(index_name))
        # logging.info("re-indexing {}".format(index_name))
        # run_index(collection_file, index_root, index_name)
        # # break
        
        # if os.path.exists(os.path.join(index_root, index_name, 'ivfpq.70.faiss')):
        #     print("faiss indexing {} exists".format(index_name))
        #     logging.info("faiss indexing {} exists".format(index_name))            
        #     continue
        
        # print("rerun faiss indexing {}".format(index_name))
        # logging.info("rerun faiss indexing {}".format(index_name))
        # run_faiss_index(index_root, index_name)
        
        
        ##### rerun retrieval   for 79 repos
        print("rerun retrieving {}".format(index_name))
        logging.info("rerun retrieving {}".format(index_name))
        queries_file = os.path.join(QUERY_DIR, index_name + ".tsv")
        run_retrieval(queries_file, index_root, index_name, "100", "commits_train")
        
        
        
    