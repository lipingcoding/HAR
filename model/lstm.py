import pdb
import pickle
from typing import SupportsAbs
from numpy.core.numeric import base_repr
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from .rgcn import ARGCN


class Model(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        # self.code_embedding = nn.Embedding(args.n_codes, args.emb_dim)
        # nn.init.uniform_(self.code_embedding.weight, -1.0, 1.0)
        
        # self.gcn = GCN(args)
        self.argcn = ARGCN(args)
        
        self.lstmcell = nn.LSTMCell(args.emb_dim, args.emb_dim)
        self.linear = nn.Linear(args.emb_dim, args.n_class)



    def forward(self, bg_list, mask):
        ps = []
        h, c = None, None
        rel_atts, edge_atts = [], []
        for bg in bg_list:
            state, node2graph = self.prepare_state(bg, h)
            x, rel_att, edge_att = self.argcn(bg, state, node2graph)
            if h is not None:
                h, c = self.lstmcell(x, (h, c))
            else:
                h, c= self.lstmcell(x)
            p = self.linear(h)
            p = F.softmax(p, dim=-1)
            ps.append(p)
            rel_atts.append(rel_att)
            edge_atts.append(edge_att)
        
        return torch.stack(ps), rel_atts, edge_atts


    # def comp_deg_norm(self, g):
    #     # pdb.set_trace()
    #     in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    #     in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    #     norm = 1.0 / in_deg
    #     norm = norm[g.edges()[1]].unsqueeze(-1)
    #     return norm

    def prepare_state(self, bg, h):
        # pdb.set_trace()
        n_nodes = bg.batch_num_nodes().cpu().numpy()
        n_graph = len(n_nodes)
        node2graph = np.array([i for i, n in enumerate(n_nodes) for _ in range(n)]) # node_id to graph_id
        node2graph = torch.from_numpy(node2graph).cuda()
        if h is None:
            state = torch.zeros(n_graph, self.args.emb_dim).cuda()
        else:
            state = h
        return state, node2graph