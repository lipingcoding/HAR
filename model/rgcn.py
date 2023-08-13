import pdb
import pickle
from typing import SupportsAbs
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import functools
from dgl import function 
from dgl import edge_subgraph
import dgl.function as fn
from dgl.nn.functional import edge_softmax
# from dgl.nn.pytorch import RelGraphConv


class ARGCN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.code_embedding = nn.Embedding(args.n_codes, args.emb_dim)
        self.relation_embedding = nn.Embedding(args.n_rels, args.emb_dim)
        nn.init.uniform_(self.code_embedding.weight, -1.0, 1.0)
        nn.init.uniform_(self.relation_embedding.weight, -1.0, 1.0)


        self.layers = nn.ModuleList()

        n_rels = args.n_rels
        in_feats = args.emb_dim
        n_hidden = args.emb_dim
        # activation = F.relu
        if args.activation == 'relu':
            activation = F.relu
        elif args.activation == 'none':
            activation = None
        else:
            raise NotImplementedError
        Layer = RelGraphConv
        for i in range(args.n_layers):
            if i == 0:
                self.layers.append(Layer(args, in_feats, n_hidden, num_rels=n_rels, activation=activation, low_mem=True))
            else:
                self.layers.append(Layer(args, n_hidden, n_hidden, num_rels=n_rels, activation=activation, low_mem=True))
        self.dropout = nn.Dropout(p=args.dropout)

    # def forward(self, g, features, etypes, norm=None):
    def forward(self, g, state, node2graph):
        features = self.code_embedding(g.ndata['id'])

        h = features
        for _, layer in enumerate(self.layers):
            # e_feat = self.relation_embedding(etypes)
            h, rel_att, edge_att = layer(g, h, self.relation_embedding.weight, state, node2graph)
            h = self.dropout(h)

        g.ndata['emb'] = h
        x = dgl.readout_nodes(g, 'emb', weight='readout_weight', op='sum')

        return x, rel_att.cpu(), edge_att.cpu()


class RelGraphConv(nn.Module):
     
    def __init__(self,
                args, 
                 in_feat,
                 out_feat,
                 num_rels,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 low_mem=False,
                 num_heads=4,
                 negative_slope=0.2
                 ):
        super(RelGraphConv, self).__init__()
        self.args = args
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.low_mem = low_mem

        # if regularizer == "basis":
            # add basis weights
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        # message func
        self.message_func = self.basis_message_func

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)


        # weight for self loop
        # if self.self_loop:
        #     self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        #     nn.init.xavier_uniform_(self.loop_weight,
                                    # gain=nn.init.calculate_gain('relu'))

        self.fc_rel_att = nn.Linear(self.in_feat, out_feat * num_heads, bias=True)
        self.fc_edge_att = nn.Linear(self.in_feat, out_feat * num_heads, bias=True)
        self.num_heads = num_heads
        self.attn_h = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feat)))
        self.attn_t = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feat)))
        # if args.use_state:
        self.attn_s = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feat)))
        # if args.use_relation:
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feat)))
        self.attn_rs = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feat)))
        # self.feat_drop = nn.Dropout(feat_drop)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()


    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_rel_att.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge_att.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_h, gain=gain)
        nn.init.xavier_normal_(self.attn_t, gain=gain)
        if hasattr(self, 'attn_s'):
            nn.init.xavier_normal_(self.attn_s, gain=gain)
        if hasattr(self, 'attn_r'):
            nn.init.xavier_normal_(self.attn_r, gain=gain)

    def basis_message_func(self, edges, etypes):
        weight = self.weight

        h = edges.src['h']
        device = h.device

    
        if self.low_mem:
            # A more memory-friendly implementation.
            # Calculate msg @ W_r before put msg into edge.
            assert isinstance(etypes, list)
            h_t = torch.split(h, etypes)
            msg = []
            for etype in range(self.num_rels):
                if h_t[etype].shape[0] == 0:
                    continue
                msg.append(torch.matmul(h_t[etype], weight[etype]))
            msg = torch.cat(msg)
        else:
            # Use batched matmult
            if isinstance(etypes, list):
                etypes = torch.repeat_interleave(torch.arange(len(etypes), device=device),
                                              torch.tensor(etypes, device=device))
            weight = weight.index_select(0, etypes)
            msg = torch.bmm(h.unsqueeze(1), weight).squeeze(1)
        
        # if self.args.use_state:
        # else:
        #     msg = msg * edges.data['norm']
        if self.args.rel and self.args.node:
            msg = (msg * edges.data['edge_att'] * edges.data['rel_att']).sum(dim=1)
        elif self.args.rel:
            msg = (msg  * edges.data['rel_att']).sum(dim=1)
        elif self.args.node:
            msg = (msg * edges.data['edge_att']).sum(dim=1)
        else:
            raise NotImplementedError

        return {'msg': msg}

 
    def forward(self, g, feat, rel_emb, state, node2graph):

        etypes = g.edata['rel_type']

        if self.args.rel:
            self.cal_rel_att(g, rel_emb, state, node2graph) # (B, n_rel, n_heads, 1)
        if self.args.node:
            self.cal_edge_att(g, feat, state, node2graph)
        else:
            mul_head_feat = self.fc_edge_att(feat).view(-1, self.num_heads, self.out_feat)
            g.srcdata['h'] = mul_head_feat

        if isinstance(etypes, torch.Tensor):
            if len(etypes) != g.num_edges():
                raise dgl.DGLError('"etypes" tensor must have length equal to the number of edges'
                               ' in the graph. But got {} and {}.'.format(
                                   len(etypes), g.num_edges()))
            if self.low_mem:
                #  When enabled,
                # it first sorts the graph based on the edge types (the sorting will not
                # change the node IDs). It then converts the etypes tensor to an integer
                # list, where each element is the number of edges of the type.
                # Sort the graph based on the etypes
                sorted_etypes, index = torch.sort(etypes)
                g = edge_subgraph(g, index, preserve_nodes=True)
                # Create a new etypes to be an integer list of number of edges.
                pos = torch.searchsorted(sorted_etypes, torch.arange(self.num_rels, device=g.device))
                num = torch.tensor([len(etypes)], device=g.device)
                etypes = (torch.cat([pos[1:], num]) - pos).tolist()
                # if norm is not None:
                #     norm = norm[index]

        # if norm is not None:
        #     g.edata['norm'] = norm
        if self.self_loop:
            loop_message = feat
        # message passing
        g.update_all(functools.partial(self.message_func, etypes=etypes),
                        function.sum(msg='msg', out='h'))
        # apply bias and activation
        node_repr = g.dstdata['h']
        # pdb.set_trace()
        if self.bias:
            node_repr = node_repr + self.h_bias
        if self.self_loop:
            node_repr = (1-self.args.loop_lambda) * node_repr + self.args.loop_lambda * loop_message
            
        if self.activation:
            node_repr = self.activation(node_repr)
        return node_repr, g.edata['rel_att'], g.edata['edge_att'] 


    def cal_edge_att(self, graph, feat, state, node2graph):
        # linear transformation
        # node_state = feat[node2graph]
        feat = self.fc_edge_att(feat).view(-1, self.num_heads, self.out_feat)
        graph.srcdata['h'] = feat

        # FNN
        # pdb.set_trace()
        eh = (feat * self.attn_h).sum(dim=-1).unsqueeze(-1)
        et = (feat * self.attn_t).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'eh': eh})
        graph.dstdata.update({'et': et})

        # compute edge attention
        # if self.args.use_state:
        if self.args.state:
            node_state = state[node2graph[graph.nodes()]]
            state = self.fc_edge_att(node_state).view(-1, self.num_heads, self.out_feat)
            es = (state * self.attn_s).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'es': es})
            graph.apply_edges(fn.u_add_v('eh', 'et', 'eht')) # elr = el + er
            graph.apply_edges(fn.u_add_e('es', 'eht', 'e')) # e = el + er + es
        else:
            graph.apply_edges(fn.u_add_v('eh', 'et', 'e'))
        # else:
        #     graph.apply_edges(fn.u_add_v('eh', 'et', 'e'))
        # pdb.set_trace()
        # if self.args.use_relation:
        #     er = (e_feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
        #     graph.edata.update({'er': er})
        #     e = self.leaky_relu(graph.edata.pop('e') + graph.edata.pop('er'))
        # else:
        e = self.leaky_relu(graph.edata.pop('e'))
        
        # compute softmax
        # graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        # return self.attn_drop(edge_softmax(graph, e))
        graph.edata['edge_att'] = edge_softmax(graph, e)

    def cal_rel_att(self, g, rel_emb, state, node2graph):
        etypes = g.edata['rel_type']
        rel_emb = self.fc_rel_att(rel_emb).view(-1, self.num_heads, self.out_feat)
        state = self.fc_rel_att(state).view(-1, self.num_heads, self.out_feat)
        er = (rel_emb * self.attn_r).sum(dim=-1).unsqueeze(-1).unsqueeze(0)
        if self.args.state:
            
            es = (state * self.attn_rs).sum(dim=-1).unsqueeze(-1).unsqueeze(1)
            # return er + es
            ers = self.leaky_relu(er + es)
        # return F.softmax(ers, dim=1)
            # pdb.set_trace()
            rel_att = F.softmax(ers, dim=1) # (B, n_rel, n_heads, 1)
            src_nodes, tar_nodes = g.edges()
            assert (node2graph[src_nodes] == node2graph[tar_nodes]).all()
            g_ids = node2graph[src_nodes]
            g.edata['rel_att'] = rel_att[g_ids, etypes]

        else:
            er = self.leaky_relu(er)
            rel_att = F.softmax(er, dim=1).squeeze(0)
            g.edata['rel_att'] = rel_att[etypes]

        
       

        
        
        
