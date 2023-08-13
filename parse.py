import argparse
import sys
import os
from util import set_seed, Logger, mkdir_if_not_exist, nowdt

parser = argparse.ArgumentParser()
parser.add_argument('--n_codes', type=int, default=1297, help='num of cui codes in semmed kg')
parser.add_argument('--emb_dim', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--n_rels', type=int, default=18)
parser.add_argument('--eval_freq', type=int, default=5)
parser.add_argument('--dropout', type=float, default=0.5)

parser.add_argument('--edge_type', default='bi_edge', help='edge, reverse_edge or bi_edge?')
parser.add_argument('--topk', type=int, default=20)

parser.add_argument('--num_heads', type=int, default=4)

parser.add_argument('--data', default='mimic-iii', help='mimic-iii or cms?')
parser.add_argument('--cms_n', type=int)

parser.add_argument('--num_workers', type=int, default=4)

# parser.add_argument('--use_state', action='store_true', default=False)
# parser.add_argument('--use_relation', action='store_true', default=False)
parser.add_argument('--eval_epoch', default=-1, type=int)

parser.add_argument('--loop_lambda', type=float, default=0.5)
parser.add_argument('--activation', default='relu')

parser.add_argument('--start_epoch', type=int, default=0)

parser.add_argument('--state', action='store_true', default=False, help='whether to use state in calculate attention')
parser.add_argument('--rel', action='store_true', default=False, help='whether to use relation att')
parser.add_argument('--node', action='store_true', default=False, help='whether to use node att')

args = parser.parse_args()

if args.data == 'mimic-iii':
    args.n_class = 1086
elif args.data == 'mimic-iv':
    args.n_class = 1271
else: 
    if args.cms_n == 1:
        args.n_class = 1159
    elif args.cms_n == 20:
        args.n_class = 1278


set_seed(args.seed)

args.log_dir = f'log/har_state_{args.state}_rel_{args.rel}_node_{args.node}_lstm_{args.data}_epoch_{args.n_epochs}_layer_{args.n_layers}_lambda_{args.loop_lambda}_act_{args.activation}_dim_{args.emb_dim}_drop_{args.dropout}_head_{args.num_heads}_lr_{args.lr}_batch_{args.batch_size}_seed_{args.seed}_eval_{args.eval_freq}_top_{args.topk}'
log_file_path = os.path.join(args.log_dir, 'log.txt')

mkdir_if_not_exist(log_file_path)
sys.stdout = Logger(log_file_path)

print(nowdt())
print(args)