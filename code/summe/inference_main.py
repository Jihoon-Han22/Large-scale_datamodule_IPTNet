import os
import random
import numpy as np
from argparse import ArgumentParser
import yaml, json
from ortools.algorithms.pywrapknapsack_solver import KnapsackSolver
import torch.backends.cudnn
from inference import inference
from helper import Average, update_args


parser = ArgumentParser()

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--data_dir', type=str, default='datasets/')
parser.add_argument('--splits', type=str, required=True)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--loc_lr', type=float, default=0.0005)
parser.add_argument('--clip_norm', type=float, default=1.0)
parser.add_argument('--model_dir', type=str, default='results/')
parser.add_argument('--tags', type=str, default='dev')
parser.add_argument('--max_epoch', type=int, default=1000)
parser.add_argument('--nms_thresh', type=float, default=0.4)
parser.add_argument('--num_feature', type=int, default=1024)
parser.add_argument('--num_hidden', type=int, default=128)
parser.add_argument('--depth', type=int, default=9)
parser.add_argument('--num_head', type=int, default=8)
parser.add_argument('--hidden_dim', type=int, default=1024)
parser.add_argument('--mlp_ratio', type=int, default=4)
parser.add_argument('--ema', type=float, default=0.3)
parser.add_argument('--set_epoch', type=int, default=1)
parser.add_argument('--coll_step', type=int, default=100)
parser.add_argument('--scan_range', type=int, default=10)
parser.add_argument('--seq_blocks', type=bool, default=True)
parser.add_argument('--max_pos_len', type=int, default=128)
parser.add_argument('--num_words', type=int, default=None)
parser.add_argument('--num_chars', type=int, default=None)
parser.add_argument('--eval_period', type=int, default=10)
parser.add_argument('--word_dim', type=int, default=300)
parser.add_argument('--char_dim', type=int, default=50)
parser.add_argument('--visual_dim', type=int, default=1024)
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--loc_heads', type=int, default=8)
parser.add_argument('--drop_rate', type=float, default=0.2)
parser.add_argument('--loc_lambda', type=float, default=5.0)
parser.add_argument('--total_step', type=int, default=6000)
parser.add_argument('--period', type=int, default=100)


def main():

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    output_path = os.path.join(args.model_dir, args.tags)
    check_point_path = os.path.join(output_path, 'checkpoints')
    log_path = os.path.join(output_path, 'logs')

    with open(args.splits, 'r') as fd:
        splits = json.load(fd)
        
    args = update_args(args)

    split_name = args.splits.split('/')[-1].split('.')[0]

    for index, split in enumerate(splits):
        inference(args, index, split, os.path.join(check_point_path, f'{split_name}_{index}.pt'), os.path.join(log_path, f'{split_name}_{index}'))






if __name__ == '__main__':
    main()

