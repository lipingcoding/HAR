from operator import mod
import numpy as np
import pdb
from sklearn.metrics._plot.precision_recall_curve import PrecisionRecallDisplay
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
import pickle
import random
import dgl
from collections import Counter

from load_mimic import load_mimic, merge_graph_list, load_cms
from parse import args
from model import Model, Loss
from util import print_total_params

import os


# def build_semmed_kg(args):
#     # with open('data/semmed/edges.pkl', 'rb') as f:
#     #     edges = pickle.load(f)

#     with open('data/semmed/edges.pkl', 'rb') as f:
#         edge_dict = pickle.load(f)

#     edges = edge_dict[args.edge_type]

#     src, tar = [], []
#     rel = []
#     for (s, t), rels in edges.items():
#         src.append(s)
#         tar.append(t)
#         rel.append(random.choice(list(rels)))

#     kg = dgl.graph((src, tar), num_nodes=args.n_codes)
#     kg.ndata['id'] = torch.arange(kg.num_nodes())
#     kg.edata['rel_type'] = torch.LongTensor(rel)

#     print(f'{len(set(src)|set(tar))} codes in semmed kg')

#     return kg

def build_semmed_kg(args):
    # with open('data/semmed/bi_edges.pkl', 'rb') as f:
    #     edges = pickle.load(f)
    with open('data/semmed/edges.pkl', 'rb') as f:
        edge_dict = pickle.load(f)

    edges = edge_dict[args.edge_type]

    src, tar = [], []
    rel = []
    for (s, t), rels in edges.items():
        rels = list(rels)
        # src.append(s)
        # tar.append(t)
        # rel.append(random.choice(list(rels)))
        src += [s for _ in rels]
        tar += [t for _ in rels]
        rel += [r for r in rels] 

    kg = dgl.graph((src, tar), num_nodes=args.n_codes)
    kg.ndata['id'] = torch.arange(kg.num_nodes())
    kg.edata['rel_type'] = torch.LongTensor(rel)

    print(f'{len(set(src)|set(tar))} codes in semmed kg')

    return kg




def topk_acc(y, p, k, y_grouped):
    total_counter = Counter()
    correct_counter = Counter()
    
    for i in range(y.shape[0]):
        true_labels = np.nonzero(y[i, :])[0]
        predictions = np.argsort(p[i, :])[-k:]
        for l in true_labels:
            total_counter[l] += 1
            correct_counter[l] += np.in1d(l, predictions, assume_unique=True).sum()

    y_grouped = args.grouped_y
    n_groups = len(y_grouped)
    total_labels = [0] * n_groups
    correct_labels = [0] * n_groups
    for i, group in enumerate(y_grouped):
        for l in group:
            correct_labels[i] += correct_counter[l]
            total_labels[i] += total_counter[l]

    acc_at_k_grouped = [x/float(y) for x, y in zip(correct_labels, total_labels)]
    acc_at_k = sum(correct_labels) / float(sum(total_labels))
    # print(f'acc at {args.topk} {acc_at_k} {str(acc_at_k_grouped)}')

    return acc_at_k, acc_at_k_grouped


def visit_level_precision(y, p, mask, k):
    # n_correct = 0
    ret_lst = []
    for i in range(y.shape[0]):
        predictions = np.argsort(p[i, :])[-k:]
        true_labels = np.nonzero(y[i, :])[0]
        n_correct = np.in1d(true_labels, predictions, assume_unique=True).sum()
        # pdb.set_trace()
        if mask[i] > 0:
            assert len(true_labels) > 0
            ret_lst.append(n_correct / min(len(true_labels), k))
        
    return np.mean(ret_lst)



def train(model, train_loader, val_loader, test_loader, args):
    if args.eval_epoch > 0:
        state_path = os.path.join(args.log_dir, f'model_state_{args.eval_epoch}.pth')
        print(f'Loading {state_path}')
        with open(state_path, 'rb') as f:
            # torch.load(f)
            model.load_state_dict(torch.load(f))

            test_acc, test_acc_grouped, test_precisions = eval(model, test_loader, args)

            for k in sorted(test_acc.keys()):
                print(f'k = {k}: visit-level precision {test_precisions[k]:.4f}')
                print(f'k = {k}: code-level accuracy {test_acc[k]:.4f}')
                print(f'Grouped test acc {test_acc_grouped[k][0]:.4f}, {test_acc_grouped[k][1]:.4f}, {test_acc_grouped[k][2]:.4f}, {test_acc_grouped[k][3]:.4f}, {test_acc_grouped[k][4]:.4f}')
        return 0

        
    loss_func = Loss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    best_val_acc, choosed_test_acc, choosed_epoch = -999999, -999999, -1

    if args.start_epoch > 0:
        state_path = os.path.join(args.log_dir, f'model_state_{args.start_epoch-1}.pth')
        print(f'Loading state_dict from {state_path}')
        with open(state_path, 'rb') as f:
            model.load_state_dict(torch.load(f))

    for epoch in tqdm(range(args.start_epoch, args.n_epochs)):
    # for epoch in tqdm(range(args.n_epochs)):

    
        model.train()
        losses = []
        # for bg, y, mask in tqdm(train_loader):
        #     optimizer.zero_grad()
        #     y, mask = y.cuda(), mask.cuda()
        #     p = model(bg.to(torch.device('cuda')), mask)
        #     loss = loss_func(p, y, mask)
        #     loss.backward()
        #     optimizer.step()
        #     losses.append(loss.item())

        for bg_list, y, mask in tqdm(train_loader):
            optimizer.zero_grad()
            y, mask = y.cuda(), mask.cuda()
            bg_list = [g.to(0) for g in bg_list]
            # pdb.set_trace()
            p, _, _ = model(bg_list, mask)
            loss = loss_func(p, y, mask)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f'Epoch {epoch:04d} Loss {np.mean(losses):.4f}')
        # if epoch % args.eval_freq == 0:
        #     train_acc, train_acc_grouped = eval(model, train_loader, args)
        #     val_acc , val_acc_grouped = eval(model, val_loader, args)
        #     test_acc, test_acc_grouped = eval(model, test_loader, args)
        #     if val_acc > best_val_acc:
        #         best_val_acc = val_acc
        #         choosed_epoch = epoch
        #         choosed_test_acc = test_acc
        #         improved = '*'
        #     else:
        #         improved = ''
        #     print(f'Epoch {epoch:04d} train acc {train_acc:.4f}, val acc {val_acc:.4f}, test acc {test_acc:.4f} {improved}')
        #     print(f'Grouped train acc {str(train_acc_grouped)}\n val acc {str(val_acc_grouped)}\n test acc {str(test_acc_grouped)}')
        #     print(f'Choosed epoch {choosed_epoch:04d}, best val acc {best_val_acc:.4f}, choosed test acc {choosed_test_acc:.4f}')

        if epoch % args.eval_freq == 0:
            # train_acc, train_acc_grouped = eval(model, train_loader, args)
            val_acc , val_acc_grouped, _ = eval(model, val_loader, args)
            test_acc, test_acc_grouped, test_precisions = eval(model, test_loader, args)
            if val_acc[args.topk] > best_val_acc:
                best_val_acc = val_acc[args.topk]
                choosed_epoch = epoch
                choosed_test_acc = test_acc[args.topk]
                improved = '*'
            else:
                improved = ''
            print(f'Epoch {epoch:04d} val acc {val_acc[args.topk]:.4f}, test acc {test_acc[args.topk]:.4f} {improved}')
            print(f'Choosed epoch {choosed_epoch:04d}, best val acc {best_val_acc:.4f}, choosed test acc {choosed_test_acc:.4f}')
            for k in sorted(test_acc.keys()):
                print(f'k = {k}: visit-level precision {test_precisions[k]:.4f}')
                print(f'k = {k}: code-level accuracy {test_acc[k]:.4f}')
                print(f'Grouped test acc {test_acc_grouped[k][0]:.4f}, {test_acc_grouped[k][1]:.4f}, {test_acc_grouped[k][2]:.4f}, {test_acc_grouped[k][3]:.4f}, {test_acc_grouped[k][4]:.4f}')

            print('\n\n\n')

            state_path = os.path.join(args.log_dir, f'model_state_{epoch}.pth')
            with open(state_path, 'wb') as f:
                torch.save(model.state_dict(), f)



def eval(model, loader, args):
    model.eval()
    ys = []
    ps = []
    masks = []

    with torch.no_grad():
        # for bg, y, mask in loader:
        #     y, mask = y.cuda(), mask.cuda()
        #     p = model(bg.to(torch.device('cuda')), mask)

        #     dim = y.shape[-1]
        #     y = (y.detach().cpu().numpy()).reshape(-1, dim)
        #     ys.append(y)
        #     p = ((p * mask.unsqueeze(-1)).detach().cpu().numpy()).reshape(-1, dim)
        #     ps.append(p)

        for bg_list, y, mask in tqdm(loader):
            y, mask = y.cuda(), mask.cuda()
            bg_list = [g.to(0) for g in bg_list]
            p, _, _ = model(bg_list, mask)

            masks.append(mask.detach().cpu().numpy().reshape(-1))
            dim = y.shape[-1]
            y = (y.detach().cpu().numpy()).reshape(-1, dim)
            ys.append(y)
            p = ((p * mask.unsqueeze(-1)).detach().cpu().numpy()).reshape(-1, dim)
            ps.append(p)
        ys = np.concatenate(ys)
        ps = np.concatenate(ps)
        masks = np.concatenate(masks)

    ks = [1, 3, 5, 10, 15, 20, 25, 30]
    acc_at_ks, acc_at_ks_grouped = dict(), dict()
    precision_at_ks = dict()
    for k in ks:
        acc_at_k, acc_at_k_grouped = topk_acc(ys, ps, k, y_grouped=args.grouped_y)
        acc_at_ks[k] = acc_at_k
        acc_at_ks_grouped[k] = acc_at_k_grouped
        precision_at_ks[k] = visit_level_precision(ys, ps, masks, k)

    return acc_at_ks, acc_at_ks_grouped, precision_at_ks



if __name__ == '__main__':
    # train_set, val_set, test_set, grouped_y = load_mimic('data/mimic-iii/seq.pkl')
    if args.data == 'mimic-iii':
        id2cls = pickle.load(open('data/mimic-iii/id2cls.pkl', 'rb'))
        train_set, val_set, test_set, grouped_y = load_mimic('data/mimic-iii/seq.pkl', args.data, id2cls)
    elif args.data == 'mimic-iv':
        id2cls = pickle.load(open('data/mimic-iv/id2cls.pkl', 'rb'))
        train_set, val_set, test_set, grouped_y = load_mimic('data/mimic-iv/seq.pkl', args.data, id2cls)
    else:
        assert args.data == 'cms'
        train_set, val_set, test_set, grouped_y = load_cms(f'data/cms/seq_{args.cms_n}.pkl', args.cms_n)
        id2cls = pickle.load(open(f'data/cms/id2cls_{args.cms_n}.pkl', 'rb'))
    args.grouped_y = grouped_y
    kg = build_semmed_kg(args)
    # collate_fn = merge_graphs(kg, id2cls, args)
    collate_fn = merge_graph_list(kg, id2cls, args)
    train_loader = DataLoader(train_set, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, collate_fn=collate_fn, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, collate_fn=collate_fn, batch_size=args.batch_size, num_workers=args.num_workers)

    # args.num_class = args.n_codes
    model = Model(args).cuda()
    train(model, train_loader, val_loader, test_loader, args)
    # eval(model, train_loader, args)