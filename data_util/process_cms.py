import numpy as np
from pandas import read_csv
import pickle
import pdb
import os
import argparse

from path_def import icd2cui_path, cms_path, cui2id_path
from util import parse_as_icd9


def read_file(csv_path):
    # read csv file
    df = read_csv(csv_path)

    df = df[['DESYNPUF_ID', 'ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2', 'ICD9_DGNS_CD_3', 'ICD9_DGNS_CD_4',
        'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6', 'ICD9_DGNS_CD_7', 'ICD9_DGNS_CD_8',
        'ICD9_DGNS_CD_9', 'ICD9_DGNS_CD_10']]
    df_list = df.values.tolist()
    df['row_index'] = range(len(df))

    first_index = df.groupby('DESYNPUF_ID').first()['row_index'].values
    last_index = df.groupby('DESYNPUF_ID').last()['row_index'].values
    assert len(first_index) == len(last_index)
    index_range = [(start, end + 1) for start, end in zip(first_index, last_index)]

    lst_of_lst_of_rows = [df_list[start: end] for (start, end) in index_range]

    lst_of_array = [np.array(lst)[:, 1:] for lst in lst_of_lst_of_rows ] # convert to array and remove the first column

    return lst_of_array


def process(lst_of_array, n):
    with open(icd2cui_path, 'rb') as f:
        icd2cui = pickle.load(f)
    with open(cui2id_path, 'rb') as f:
        cui2id = pickle.load(f)


    # filtering 
    icd9s = set()
    cuis = set()
    cnt_icd9 = 0
    cnt_cui = 0
    cnt_selected = 0

    types = {}
    newSeqs = []

    filtered = set()
    missed = set()
    filtered_occurrence = 0
    missed_occurrence = 0

    for patient in lst_of_array:
        newPatient = []
        for visit in patient:
            newVisit = []
            for icd in visit:
                # pdb.set_trace()
                if icd == 'nan':
                    continue
                icd = parse_as_icd9(icd)
                icd9s.add(icd)
                cnt_icd9 += 1
                if icd in icd2cui:
                    cui = icd2cui[icd]
                    cuis.add(cui)
                    cnt_cui += 1
                    if cui in cui2id:
                        id = cui2id[cui]
                        newVisit.append(id)
                        cnt_selected += 1
            if len(newVisit) > 0:
                newPatient.append(newVisit)
        if len(newPatient) > 1:
            newSeqs.append(newPatient)

    id2cls = {}
    for patient in newSeqs:
        for visit in patient:
            for id in visit:
                if id not in id2cls:
                    id2cls[id] = len(id2cls)
    print(f'There are originally {len(icd9s)} icd9 in cms')
    print(f'There are {len(cuis)} cuis')
    print(f'Number of unique codes selected for cms {len(id2cls)}')
    print(f'Occurrences {cnt_icd9} {cnt_cui} {cnt_selected}')
    cms_cui_seq_path = os.path.join(cms_path, f'seq_{n}.pkl')
    cms_id2cls_path = os.path.join(cms_path, f'id2cls_{n}.pkl')
    pickle.dump(newSeqs, open(cms_cui_seq_path, 'wb'), -1)
    pickle.dump(id2cls, open(cms_id2cls_path, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int)
    args = parser.parse_args()

    final_lst_of_array = None
    for i in range(args.n):
        csv_path = f'../data/cms/DE1_0_2008_to_2010_Inpatient_Claims_Sample_{i+1}.csv'
        lst_of_array = read_file(csv_path)
        if final_lst_of_array is None:
            final_lst_of_array = lst_of_array
        else:
            final_lst_of_array += lst_of_array

    process(final_lst_of_array, args.n)
            

