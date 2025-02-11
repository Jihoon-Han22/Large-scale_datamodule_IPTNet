import os
import random
import numpy as np
from argparse import ArgumentParser
import json
# from ortools.algorithms.pywrapknapsack_solver import KnapsackSolver
import torch.backends.cudnn
# from sum_train import train
from sum_inference import train
from helper import Average, update_args


parser = ArgumentParser()

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--data_dir', type=str, default='../../kwanseok/iPTNet/datasets/')
parser.add_argument('--splits', type=str, required=True)
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--loc_lr', type=float, default=0.0005)
parser.add_argument('--clip_norm', type=float, default=1.0)
parser.add_argument('--model_dir', type=str, default='results/')
parser.add_argument('--tags', type=str, default='dev')
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--nms_thresh', type=float, default=0.4)
parser.add_argument('--num_feature', type=int, default=1024)
parser.add_argument('--num_hidden', type=int, default=128) # 128
parser.add_argument('--depth', type=int, default=9) # 9
parser.add_argument('--num_head', type=int, default=8)
parser.add_argument('--hidden_dim', type=int, default=1024) #1024
parser.add_argument('--mlp_ratio', type=int, default=1) # 4
parser.add_argument('--ema', type=float, default=0.3)
parser.add_argument('--set_epoch', type=int, default=1)
parser.add_argument('--coll_step', type=int, default=100)
parser.add_argument('--scan_range', type=int, default=10)
parser.add_argument('--seq_blocks', type=bool, default=True) # True
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
parser.add_argument('--period', type=int, default=10)




def main():

    args = parser.parse_args()

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    output_path = os.path.join(args.model_dir, args.tags)
    check_point_path = os.path.join(output_path, 'checkpoints')
    log_path = os.path.join(output_path, 'logs')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(check_point_path):
        os.makedirs(check_point_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # with open(args.splits) as f:
    #     splits = yaml.safe_load(f)

    with open(args.splits, 'r') as fd:
        splits = json.load(fd)
        
    args = update_args(args)

    stats = Average('fscore')
    test_stats = Average('test_fscore')

    split_name = args.splits.split('/')[-1].split('.')[0]
    index = 0
    print(f'Model is training on MR.Sum dataset.')
    test_metric, final_map50, final_map15, final_sRho, final_kTau, final_WIR, final_WSE, final_IR, final_CIS = train(args, index, splits, check_point_path, os.path.join(log_path, f'{split_name}_{index}'))
    # stats.update(fscore=metric)
    print(f'Test score is : {test_metric:.4f}%')
    print(f'Final map50 is : {final_map50:.4f}%')
    print(f'Final map15 is : {final_map15:.4f}%')
    print(f'Final sRho is : {final_sRho:.4f}%')
    print(f'Final kTau is : {final_kTau:.4f}%')
    print(f'Final WIR is : {final_WIR:.4f}%')
    print(f'Final WSE is : {final_WSE:.4f}%')
    print(f'Final IR is : {final_IR:.4f}%')
    print(f'Final CIS is : {final_CIS:.4f}%')
    
    f = open(os.path.join(output_path, 'results.txt'), 'a')
    f.write("Testing on Model " + str(output_path) + '\n')
    f.write('Test F-score ' + str(test_metric) + '\n')
    f.write('Test MAP50   ' + str(final_map50) + '\n')
    f.write('Test MAP15   ' + str(final_map15) + '\n\n')
    f.flush()
    #test_stats.update(test_fscore=test_metric)
    # split = splits[3]
    # index=3
    # print(f'Model is training on split {index}.')
    # metric, test_metric = train(args, index, split, os.path.join(check_point_path, f'{split_name}_{index}.pt'), os.path.join(log_path, f'{split_name}_{index}'))
    # stats.update(fscore=metric)
    # print(f'Test score at split {index} : {test_metric:.4f}%')
    # test_stats.update(test_fscore=test_metric)

    # print(f'Model has completed the training. F-score is: {stats.fscore}.')
    # print(f'Model has completed the training. F-score is: {test_stats.test_fscore}.')





if __name__ == '__main__':
    main()

