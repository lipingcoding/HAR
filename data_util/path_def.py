import os


DATA_DIR =  '../data'

# for simplify_semmed use
ori_semmed_path = os.path.join(DATA_DIR, 'semmed/semmed.csv')
simplified_semmed_path = os.path.join(DATA_DIR, 'semmed/database.csv')

output_edges_path = os.path.join(DATA_DIR, 'semmed/edges.pkl')
output_graph_path = os.path.join(DATA_DIR, 'semmed/kg.pkl')
output_bi_edges_path = os.path.join(DATA_DIR, 'semmed/bi_edges.pkl')

cui2id_path = os.path.join(DATA_DIR, 'semmed/cui2id.pkl')


# for prepare icd2cui use
snomedct_path = os.path.join(DATA_DIR, 'snomed/SNOMEDCT_CORE_SUBSET_202105.txt')
icd2snomed_1to1_path = os.path.join(DATA_DIR, 'snomed/ICD9CM_SNOMED_MAP_1TO1_202012.txt')
icd2snomed_1toM_path = os.path.join(DATA_DIR, 'snomed/ICD9CM_SNOMED_MAP_1TOM_202012.txt')
icd2cui_path = os.path.join(DATA_DIR, 'snomed/icd2cui.pkl')

# for process mimic use
mimic_path = os.path.join(DATA_DIR, 'mimic-iii')
mimic_cui_seq_path = os.path.join(mimic_path, 'seq.pkl')
mimic_id2cls_path = os.path.join(mimic_path, 'id2cls.pkl')

# for process mimic-iv use
mimic_iv_path = os.path.join(DATA_DIR, 'mimic-iv')
mimic_iv_cui_seq_path = os.path.join(mimic_iv_path, 'seq.pkl')
mimic_iv_id2cls_path = os.path.join(mimic_iv_path, 'id2cls.pkl')

# for process cms use
cms_path = os.path.join(DATA_DIR, 'cms')
cms_cui_seq_path = os.path.join(cms_path, 'seq.pkl')
cms_id2cls_path = os.path.join(cms_path, 'id2cls.pkl')

