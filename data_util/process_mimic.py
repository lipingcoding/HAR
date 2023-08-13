import os
from datetime import datetime
import pickle

from path_def import mimic_path, icd2cui_path, mimic_cui_seq_path, cui2id_path, mimic_id2cls_path
from util import parse_as_icd9


if __name__ == '__main__':
    admissionFile = os.path.join(mimic_path, 'ADMISSIONS.csv')
    diagnosisFile = os.path.join(mimic_path, 'DIAGNOSES_ICD.csv')
    # outFile = os.path.join(mimic_path, 'processed')

    print('Building pid-admission mapping, admission-date mapping')
    pidAdmMap = {}
    admDateMap = {}
    infd = open(admissionFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        admId = int(tokens[2])
        admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
        admDateMap[admId] = admTime
        if pid in pidAdmMap: pidAdmMap[pid].append(admId)
        else: pidAdmMap[pid] = [admId]
    infd.close()

    print('Building admission-dxList mapping')
    admDxMap = {}
    infd = open(diagnosisFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[2])
        dxStr = parse_as_icd9(tokens[4][1:-1]) ############## Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.

        if admId in admDxMap: 
            admDxMap[admId].append(dxStr)
        else: 
            admDxMap[admId] = [dxStr]

    infd.close()

    print('Building pid-sortedVisits mapping')
    pidSeqMap = {}
    for pid, admIdList in pidAdmMap.items():
        if len(admIdList) < 2: continue

        sortedList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in admIdList])
        pidSeqMap[pid] = sortedList

    
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

    icd9s = set()
    cuis = set()
    cnt_icd9 = 0
    cnt_cui = 0
    cnt_selected = 0
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for icd in visit:
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
    print(f'There are {len(newSeqs)} patients')
    print(f'There are {n_visit} visits')
    print(f'Max visits per patient {max_len_visit}')
    print(f'Average codes per visit {n_occurrence}/{n_visit} = {n_occurrence/n_visit}')
    print(f'Max number of codes {max_n_code}')
    print(f'There are originally {len(icd9s)} icd9 in mimic')
    print(f'There are {len(cuis)} cuis')
    print(f'Number of unique codes selected for mimic {len(id2cls)}')
    print(f'Occurrences {cnt_icd9} {cnt_cui} {cnt_selected}')
    pickle.dump(newSeqs, open(mimic_cui_seq_path, 'wb'), -1)
    pickle.dump(id2cls, open(mimic_id2cls_path, 'wb'))
