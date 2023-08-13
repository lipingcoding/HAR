import pickle
import numpy as np
import pdb
import torch
from time import sleep, time
import dgl
from dgl.sampling import sample_neighbors
import os

from torch.utils.data import Dataset, DataLoader

_TEST_RATIO = 0.2
_VALIDATION_RATIO = 0.1


class EHRDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index: int):
        return self.data[index]


    def __len__(self) -> int:
        return len(self.data)


def padMatrix(seqs):
    labels = seqs
    lengths = np.array([len(seq) for seq in seqs]) - 1
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    num_class=1717

    x = np.zeros((maxlen, n_samples, num_class))
    y = np.zeros((maxlen, n_samples, num_class))
    mask = np.zeros((maxlen, n_samples))


    for idx, (seq, lseq) in enumerate(zip(seqs, labels)):
        for xvec, subseq in zip(x[:,idx,:], seq[:-1]): xvec[subseq] = 1.
        for yvec, subseq in zip(y[:,idx,:], lseq[1:]): yvec[subseq] = 1.
        mask[:lengths[idx], idx] = 1 # 后面用零填充

    # lengths = np.array(lengths)

    # return x, y, mask, lengths
    return torch.FloatTensor(x), torch.FloatTensor(y), torch.LongTensor(mask)


def upto_k_neighbor_nodes(g, seed_nodes, k):
    for _ in range(k):
        in_nodes = list(torch.cat(sample_neighbors(g, seed_nodes, fanout=-1, edge_dir='in').edges()).numpy())
        out_nodes = list(torch.cat(sample_neighbors(g, seed_nodes, fanout=-1, edge_dir='out').edges()).numpy())
        new_nodes = set(in_nodes + out_nodes)
        seed_nodes = list(new_nodes | set(seed_nodes))
    
    return seed_nodes

# def load_mimic(seq_path):
#     split_path = 'data/mimic-iii/split.pkl'
class merge_graphs():
    def __init__(self, kg, id2cls, args) -> None:
        self.kg = kg
        self.id2cls = id2cls
        self.num_class = args.n_class

    def __call__(self, seqs):
        # t0 = time()
        labels = seqs
        lengths = np.array([len(seq) for seq in seqs]) - 1
        n_samples = len(seqs)
        maxlen = np.max(lengths)
        
        num_class = self.num_class
        
        y = np.zeros((maxlen, n_samples, num_class))
        mask = np.zeros((maxlen, n_samples))

        for idx, (seq, lseq) in enumerate(zip(seqs, labels)):
            for yvec, subseq in zip(y[:,idx,:], lseq[1:]): 
                # yvec[subseq] = 1.
                cls = [self.id2cls[id] for id in subseq]
                yvec[cls] = 1.
            mask[:lengths[idx], idx] = 1 # 后面用零填充

        graphs = []
        for patient in seqs:
            for visit in patient[:-1]:
                g = dgl.node_subgraph(self.kg, visit)
                graphs.append(g)
                
            # pad with null graphs
            for i in range(len(patient)-1, maxlen):
                g = dgl.node_subgraph(self.kg, [])
                graphs.append(g)
        
        big_graph = dgl.batch(graphs)
        # print(f'use {time()-t0} s')
        return big_graph, torch.FloatTensor(y), torch.LongTensor(mask)

def load_mimic(seq_path, data, id2cls):
    split_path = f'data/{data}/split.pkl'
    sequences = np.array(pickle.load(open(seq_path, 'rb')))
    if os.path.exists(split_path):
        print(f'Loading data split from {split_path}')
        (train_indices, valid_indices, test_indices) = pickle.load(open(split_path, 'rb'))
    else:
        

        np.random.seed(0)
        dataSize = len(sequences)

        ind = np.random.permutation(dataSize)

        nTest = int(_TEST_RATIO * dataSize)
        nValid = int(_VALIDATION_RATIO * dataSize)

        test_indices = ind[:nTest]
        valid_indices = ind[nTest:nTest+nValid]
        train_indices = ind[nTest+nValid:]

        print(f'Dumping data split to {split_path}')
        pickle.dump((train_indices, valid_indices, test_indices), open(split_path, 'wb'))

    train_set_x = sequences[train_indices]
    test_set_x = sequences[test_indices]
    valid_set_x = sequences[valid_indices]
       
    

    return EHRDataset(train_set_x), EHRDataset(valid_set_x), EHRDataset(test_set_x), group_y(train_set_x, id2cls)


def group_y(y, id2cls, num_bins=5):
    y = [id2cls[y3] for y1 in y for y2 in y1 for y3 in y2]  # flatten y
    unique, counts = np.unique(y, return_counts=True)
    total_counts = counts.sum()
    percentiles = np.linspace(0, 1, num_bins+1)[1:]
    cuts = np.ceil(percentiles * total_counts)
    y_dict = dict(zip(unique, counts))
    sorted_unique = sorted(unique, key=lambda x: y_dict[x])
    count = 0
    y_grouped = []
    cur_group = []
    group_id = 0
    for y in sorted_unique:
        cur_group.append(y)
        count += y_dict[y]
        if count > cuts[group_id]:
            y_grouped.append(cur_group)
            cur_group = []
            group_id += 1
    y_grouped.append(cur_group)
    print('Label frequencies in each group:')
    print([sum([y_dict[x] for x in y]) for y in y_grouped])

    return y_grouped


class merge_graph_list():
    def __init__(self, kg, id2cls, args) -> None:
        self.kg = kg
        self.id2cls = id2cls
        self.num_class = args.n_class
        self.n_layers = args.n_layers

    def __call__(self, seqs):
        # t0 = time()
        labels = seqs
        lengths = np.array([len(seq) for seq in seqs]) - 1
        n_samples = len(seqs)
        maxlen = np.max(lengths)
        
        num_class = self.num_class
        
        y = np.zeros((maxlen, n_samples, num_class))
        mask = np.zeros((maxlen, n_samples))

        for idx, (seq, lseq) in enumerate(zip(seqs, labels)):
            for yvec, subseq in zip(y[:,idx,:], lseq[1:]): 
                # yvec[subseq] = 1.
                cls = [self.id2cls[id] for id in subseq]
                yvec[cls] = 1.
            mask[:lengths[idx], idx] = 1 # 后面用零填充

        g_list = []
        # for patient in seqs:
        #     graphs = []
        #     for visit in patient[:-1]:
        #         # pdb.set_trace()
        #         g = dgl.node_subgraph(self.kg, self.kg.nodes())
        #         readout_weight = torch.zeros_like(g.ndata['id'])
        #         readout_weight[visit] = 1.
        #         g.ndata['readout_weight'] = readout_weight.unsqueeze(-1)
        #         graphs.append(g)
                
        #     # pad with null graphs
        #     for i in range(len(patient)-1, maxlen):
        #         g = dgl.node_subgraph(self.kg, [])
        #         graphs.append(g)
        #     g_list.append(dgl.batch(graphs))
        for i in range(maxlen):
            graphs = []
            for patient in seqs:
                # visit = patient[i]
                if i < len(patient) - 1:
                    visit = patient[i]
                    # g = dgl.node_subgraph(self.kg, self.kg.nodes())
                    # pdb.set_trace()
                    readout_weight = torch.zeros_like(self.kg.ndata['id'])
                    readout_weight[visit] = 1.
                    self.kg.ndata['readout_weight'] = readout_weight.unsqueeze(-1)
                    seed_nodes = upto_k_neighbor_nodes(self.kg, visit, self.n_layers)
                    g = dgl.node_subgraph(self.kg, seed_nodes)
                    # readout_weight = torch.zeros_like(g.ndata['id'])
                    # readout_weight[visit] = 1.
                    # readout_weight = torch.ones_like(g.ndata['id'])
                    # g.ndata['readout_weight'] = readout_weight.unsqueeze(-1)
                    graphs.append(g)
                else:
                    g = dgl.node_subgraph(self.kg, [])
                    graphs.append(g)
            g_list.append(dgl.batch(graphs))
        
        # big_graph = dgl.batch(graphs)
        # print(f'use {time()-t0} s')
        return g_list, torch.FloatTensor(y), torch.LongTensor(mask)
        
def load_cms(seq_path, n):
    split_path = f'data/cms/split_{n}.pkl'
    sequences = np.array(pickle.load(open(seq_path, 'rb')))
    if os.path.exists(split_path):
        print(f'Loading data split from {split_path}')
        (train_indices, valid_indices, test_indices) = pickle.load(open(split_path, 'rb'))
    else:
        np.random.seed(0)
        dataSize = len(sequences)

        ind = np.random.permutation(dataSize)

        nTest = int(_TEST_RATIO * dataSize)
        nValid = int(_VALIDATION_RATIO * dataSize)

        test_indices = ind[:nTest]
        valid_indices = ind[nTest:nTest+nValid]
        train_indices = ind[nTest+nValid:]

        print(f'Dumping data split to {split_path}')
        pickle.dump((train_indices, valid_indices, test_indices), open(split_path, 'wb'))

    train_set_x = sequences[train_indices]
    test_set_x = sequences[test_indices]
    valid_set_x = sequences[valid_indices]
       
    

    return EHRDataset(train_set_x), EHRDataset(valid_set_x), EHRDataset(test_set_x), group_y(train_set_x)
