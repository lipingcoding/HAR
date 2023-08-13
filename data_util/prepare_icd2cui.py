import networkx as nx
import json
from tqdm import tqdm
import codecs
import pandas as pd
import pdb
import pickle

from path_def import snomedct_path, icd2snomed_1to1_path, icd2snomed_1toM_path, icd2cui_path

def map_icd2cui(snomedct_path: str, icd2snomed_1to1_path: str, icd2snomed_1toM_path: str) -> dict:
    """
    convert icd-9 codes to umls_cui via snomed_cid

    `return`:
        icd2cui: dict: {"icd_code (str)": umls_cui (str)}
        icd_list: list of all icd codes in the icd2cui dictionary
    """
    # load snomed and icd2snomed
    snomed = pd.read_table(snomedct_path, sep="|")
    snomed2cui = snomed[["SNOMED_CID", "UMLS_CUI"]]
    icd2snomed_1to1 = pd.read_table(icd2snomed_1to1_path)
    icd2snomed_1toM = pd.read_table(icd2snomed_1toM_path)

    # convert the dataframe into dictionary
    snomed2cui_dict = snomed2cui.set_index("SNOMED_CID").T.to_dict("records")[0]  # dict: {"snomed_cid (int)": umls_cui (str)}

    # map cui to icd2snomed via snomed
    icd2snomed_1to1["UMLS_CUI"] = icd2snomed_1to1["SNOMED_CID"].map(snomed2cui_dict)
    icd2snomed_1toM["UMLS_CUI"] = icd2snomed_1toM["SNOMED_CID"].map(snomed2cui_dict)

    # drop all rows that have any NaN values
    icd2snomed_1to1 = icd2snomed_1to1.dropna(axis=0, how="any")
    icd2snomed_1toM = icd2snomed_1toM.dropna(axis=0, how="any")

    # extract icd and cui
    icd_cui_1to1 = icd2snomed_1to1[["ICD_CODE", "UMLS_CUI"]]
    icd_cui_1toM = icd2snomed_1toM[["ICD_CODE", "UMLS_CUI"]]

    # drop duplicates in icd codes
    icd_cui_1toM = icd_cui_1toM.drop_duplicates(subset=["ICD_CODE"], keep="first")

    # convert the dataframe into dictionary
    icd2cui_1to1 = icd_cui_1to1.set_index("ICD_CODE").T.to_dict("records")[0]
    icd2cui_1toM = icd_cui_1toM.set_index("ICD_CODE").T.to_dict("records")[0]
    icd2cui = {}
    icd2cui.update(icd2cui_1to1)
    icd2cui.update(icd2cui_1toM)

    # make the list of all icd codes in the dictionary
    icd_list = list(icd2cui.keys())
    cui_list = list(icd2cui.values())
    return icd2cui, icd_list, cui_list


if __name__ == "__main__":
    icd2cui, icd_list, cui_list = map_icd2cui(snomedct_path, icd2snomed_1to1_path, icd2snomed_1toM_path)
    print(f'len(icd_list) = {len(set(icd_list))}')
    print(f'len(cui_list) = {len(set(cui_list))}')
    with open(icd2cui_path, 'wb') as f:
        pickle.dump(icd2cui, f)