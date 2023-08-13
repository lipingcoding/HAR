from operator import sub
import networkx as nx
import json
from tqdm import tqdm
import codecs
import pandas as pd
import pdb
import pickle
from collections import defaultdict

from path_def import icd2cui_path, simplified_semmed_path,  output_edges_path, cui2id_path, output_bi_edges_path

relations_prune = ['affects', 'augments', 'causes', 'diagnoses', 'interacts_with', 'part_of', 'precedes', 'predisposes', 'produces']



def separate_semmed_cui(semmed_cui: str) -> list:
    """
    separate semmed cui with | by perserving the replace the numbers after |
    `param`:
        semmed_cui: single or multiple semmed_cui separated by |
    `return`:
        sep_cui_list: list of all separated semmed_cui
    """
    sep_cui_list = []
    sep = semmed_cui.split("|")
    first_cui = sep[0]
    sep_cui_list.append(first_cui)
    ncui = len(sep)
    for i in range(ncui - 1):
        last_digs = sep[i + 1]
        len_digs = len(last_digs)
        if len_digs < 8: 
            sep_cui = first_cui[:8 - len(last_digs)] + last_digs
            sep_cui_list.append(sep_cui)
    return sep_cui_list


def add_new_edge(subj, obj, rel, rev_rel, edges, reverse_edges, bi_edges, cui2id):
    if subj not in cui2id:
        cui2id[subj] = len(cui2id)
    if obj not in cui2id:
        cui2id[obj] = len(cui2id)
    subj, obj = cui2id[subj], cui2id[obj]

    edges[(subj, obj)].add(rel)
    reverse_edges[(obj, subj)].add(rel)
    bi_edges[(subj, obj)].add(rel)
    bi_edges[(obj, subj)].add(rev_rel)


def construct_subgraph(semmed_csv_path, output_edges_path, cuis):
    print("generating subgraph of SemMed using newly extracted cui list...")


    idx2relation = relations_prune
    relation2idx = {r: i for i, r in enumerate(idx2relation)}

    # graph = nx.MultiDiGraph()
    nrow = sum(1 for _ in open(semmed_csv_path, "r", encoding="utf-8"))

    edges = defaultdict(set)
    reverse_edges = defaultdict(set)
    bi_edges = defaultdict(set)
    cui2id = {}
    with open(semmed_csv_path, "r", encoding="utf-8") as fin:
        # attrs = set()
        for line in tqdm(fin, total=nrow):
            ls = line.strip().split(',')
            # pdb.set_trace()
            if ls[0].lower() not in idx2relation:
                continue
            if ls[1] == ls[2]:
                continue

            # sent = ls[1]
            rel = relation2idx[ls[0].lower()]
            rev_rel = rel + len(relation2idx)
            # pdb.set_trace()


            if ls[1].startswith("C") and ls[2].startswith("C"):
                if len(ls[1]) == 8 and len(ls[2]) == 8:
                    if ls[1] in cuis and ls[2] in cuis:
                        subj = ls[1]
                        obj = ls[2]
                        add_new_edge(subj, obj, rel, rev_rel, edges, reverse_edges, bi_edges, cui2id)
                           
                elif len(ls[1]) != 8 and len(ls[2]) == 8:
                    cui_list = separate_semmed_cui(ls[1])
                    subj_list = [s for s in cui_list if s in cuis]
                    if ls[2] in cuis:
                        obj = ls[2]
                        for subj in subj_list:
                            add_new_edge(subj, obj, rel, rev_rel, edges, reverse_edges, bi_edges, cui2id)

                          
                elif len(ls[1]) == 8 and len(ls[2]) != 8:
                    cui_list = separate_semmed_cui(ls[2])
                    obj_list = [o for o in cui_list if o in cuis]
                    if ls[1] in cuis:
                        subj = ls[1]
                        for obj in obj_list:
                            add_new_edge(subj, obj, rel, rev_rel, edges, reverse_edges, bi_edges, cui2id)
                            
                            
                else:
                    cui_list1 = separate_semmed_cui(ls[1])
                    subj_list = [s for s in cui_list1 if s in cuis]
                    cui_list2 = separate_semmed_cui(ls[2])
                    obj_list = [o for o in cui_list2 if o in cuis]
                    for subj in subj_list:
                        for obj in obj_list:
                            add_new_edge(subj, obj, rel, rev_rel, edges, reverse_edges, bi_edges, cui2id)

    edge_dict = {
        'edge': edges,
        'reverse_edge': reverse_edges,
        'bi_edge': bi_edges
    }
    with open(output_edges_path, 'wb') as f:
        pickle.dump(edge_dict, f)


    with open(cui2id_path, 'wb') as f:
        pickle.dump(cui2id, f)
    print(f'There are {len(cui2id)} unique edges in kg')
    print(f'Dump edges to {output_edges_path}')

if __name__ == "__main__":
    with open(icd2cui_path, 'rb') as f:
        icd2cui = pickle.load(f)
    cuis = set(icd2cui.values())
    
    construct_subgraph(simplified_semmed_path, output_edges_path, cuis)
    