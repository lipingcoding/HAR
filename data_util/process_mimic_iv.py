import os
from datetime import datetime
import pickle
import numpy as np
# from path_def import mimic_path, icd2cui_path, mimic_cui_seq_path, cui2id_path, mimic_id2cls_path
from util import parse_as_icd9
from path_def import mimic_iv_path, icd2cui_path, mimic_iv_cui_seq_path, cui2id_path, mimic_iv_id2cls_path


def parse_admisssion(admissionFile):
    pidAdmMap = {}
    admDateMap = {}
    infd = open(admissionFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[0]) # subject
        admId = int(tokens[1]) # hadm_id
        admTime = datetime.strptime(tokens[2], '%Y-%m-%d %H:%M:%S')

        if pid in pidAdmMap: 
            pidAdmMap[pid].append(admId)
        else: 
            pidAdmMap[pid] = [admId]

        admDateMap[admId] = admTime

    infd.close()

    print(f'In parse admission file, number of patients: {len(pidAdmMap)}')
    lengths = [len(admIdList) for pid, admIdList in pidAdmMap.items()]
    print(f'In parse admission file, average length = {np.mean(lengths):.2f}')

    return pidAdmMap, admDateMap


def parse_diagnosis(diagnosisFile):
    pidAdmMap = {}

    admDxMap = {}
    infd = open(diagnosisFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')

        pid = int(tokens[0]) # subject
        admId = int(tokens[1]) # hadm_id
        if pid in pidAdmMap and admId not in pidAdmMap[pid]:
            # pidAdmMap[pid].append(admId)
            pidAdmMap[pid].add(admId)
        else: 
            pidAdmMap[pid] = {admId}

        admId = int(tokens[1])
        dxStr = parse_as_icd9(tokens[3].strip()) ############## Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.
        # dxStr = tokens[3].strip()

        if admId in admDxMap: 
            admDxMap[admId].append(dxStr)
        else: 
            admDxMap[admId] = [dxStr]

    infd.close()

    print(f'In parse diagnosis file, number of patients: {len(pidAdmMap)}')
    lengths = [len(admIdList) for pid, admIdList in pidAdmMap.items()]
    print(f'visit2icd, average length = {np.mean(lengths):.2f}')


    return admDxMap, pidAdmMap


if __name__ == '__main__':
    admissionFile = os.path.join(mimic_iv_path, 'admissions.csv')
    pidAdmMap, admDateMap = parse_admisssion(admissionFile)

    diagnosisFile = os.path.join(mimic_iv_path, 'diagnoses_icd.csv')
    admDxMap, tmp_pidAdmMap = parse_diagnosis(diagnosisFile)

    # for patient in patients:
    #     assert patient in pidAdmMap
    for pid, adm_set in tmp_pidAdmMap.items():
        assert pid in pidAdmMap
        for adm in adm_set:
            assert adm in pidAdmMap[pid]
    

    print('Building pid-sortedVisits mapping')
    pidSeqMap = {}
    cnt_less_2 = 0
    for pid, admIdList in pidAdmMap.items():
        if len(admIdList) < 2: 
            cnt_less_2 += 1
            continue

        sortedList = sorted([(admDateMap[admId], admDxMap[admId])  for admId in admIdList if  admId in admDxMap])
        pidSeqMap[pid] = sortedList

    print(f'Patients have less than 2 visit: {cnt_less_2}')
    print(f'Filtered {len(pidSeqMap)} patients')
    
    print('Building pids, dates, strSeqs')
    pids = []
    dates = []
    seqs = []
    for pid, visits in pidSeqMap.items():
        pids.append(pid)
        seq = []
        date = []
        for visit in visits:
            date.append(visit[0])
            seq.append(visit[1])
        dates.append(date)
        seqs.append(seq)
    
    
    with open(icd2cui_path, 'rb') as f:
        icd2cui = pickle.load(f)
    with open(cui2id_path, 'rb') as f:
        cui2id = pickle.load(f)

    print('Converting strSeqs to intSeqs, and making types')
    types = {}
    newSeqs = []
    
    filtered = set()
    missed = set()
    filtered_occurrence = 0
    missed_occurrence = 0

    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for icd in visit:
                if icd in icd2cui:
                    cui = icd2cui[icd]
                    if cui in cui2id:
                        id = cui2id[cui]
                        newVisit.append(id)
            if len(newVisit) > 0:
                newPatient.append(newVisit)
        if len(newPatient) > 1:
            newSeqs.append(newPatient)

    id2cls = {}
    n_visit = 0
    max_len_visit = 0
    max_n_code = 0
    n_occurrence = 0
    for patient in newSeqs:
        n_visit += len(patient)
        max_len_visit = max(max_len_visit, len(patient))
        for visit in patient:
            n_occurrence += len(visit)
            max_n_code = max(max_n_code, len(visit))
            for id in visit:
                if id not in id2cls:
                    id2cls[id] = len(id2cls)

    print(f'Number of unique codes selected for mimic {len(id2cls)}')
    pickle.dump(newSeqs, open(mimic_iv_cui_seq_path, 'wb'), -1)
    pickle.dump(id2cls, open(mimic_iv_id2cls_path, 'wb'))

    print(f'Number of patients: {len(newSeqs)}')
    lengths = [len(seq) for seq in newSeqs]
    print(f'Average length: {np.mean(lengths):.2f}')
    print(f'Max visits per patient {max_len_visit}')

    print(f'There are {len(newSeqs)} patients')
    print(f'There are {n_visit} visits')
    print(f'Average codes per visit {n_occurrence}/{n_visit} = {n_occurrence/n_visit}')
    print(f'Max number of codes {max_n_code}')