import os
import sys
import h5py
import json
import math
import torch
import numpy as np

# from src.models.hisum_module import HiSumModule

# A Dynamic Programming based Python Program for 0-1 Knapsack problem
# Returns the maximum value that can be put in a knapsack of capacity W
import numpy as np
from ortools.algorithms import pywrapknapsack_solver


def knapsack(W, wt, val, n):
    K = [[0 for x in range(W+1)] for x in range(n+1)]

    # Build table K[][] in bottom up manner
    for i in range(n+1):
        for w in range(W+1):
            if i==0 or w==0:
                K[i][w] = 0
            elif wt[i-1] <= w:
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
            else:
                K[i][w] = K[i-1][w]


    best = K[n][W]

    amount = np.zeros(n)
    a = best
    j = n
    Y = W

    # j = j + 1;
    #
    # amount(j) = 1;
    # Y = Y - weights(j);
    # j = j - 1;
    # a = A(j + 1, Y + 1);

    while a > 0:
       while K[j][Y] == a:
           j = j - 1

       j = j + 1
       amount[j-1] = 1
       Y = Y - wt[j-1]
       j = j - 1
       a = K[j][Y]

    return amount


def test_knapsack():
    weights = [1 ,1 ,1, 1 ,2 ,2 ,3]
    values  = [1 ,1 ,2 ,3, 1, 3 ,5]
    best = 13
    print(knapsack(7, weights, values, 7))

#===========================================
'''
------------------------------------------------
Use dynamic programming (DP) to solve 0/1 knapsack problem
Time complexity: O(nW), where n is number of items and W is capacity
Author: Kaiyang Zhou
Website: https://kaiyangzhou.github.io/
------------------------------------------------
knapsack_dp(values,weights,n_items,capacity,return_all=False)
Input arguments:
  1. values: a list of numbers in either int or float, specifying the values of items
  2. weights: a list of int numbers specifying weights of items
  3. n_items: an int number indicating number of items
  4. capacity: an int number indicating the knapsack capacity
  5. return_all: whether return all info, defaulty is False (optional)
Return:
  1. picks: a list of numbers storing the positions of selected items
  2. max_val: maximum value (optional)
------------------------------------------------
'''
def knapsack_dp(values,weights,n_items,capacity,return_all=False):
    check_inputs(values,weights,n_items,capacity)

    table = np.zeros((n_items+1,capacity+1),dtype=np.float32)
    keep = np.zeros((n_items+1,capacity+1),dtype=np.float32)

    for i in range(1,n_items+1):
        for w in range(0,capacity+1):
            wi = weights[i-1] # weight of current item
            vi = values[i-1] # value of current item
            if (wi <= w) and (vi + table[i-1,w-wi] > table[i-1,w]):
                table[i,w] = vi + table[i-1,w-wi]
                keep[i,w] = 1
            else:
                table[i,w] = table[i-1,w]

    picks = []
    K = capacity

    for i in range(n_items,0,-1):
        if keep[i,K] == 1:
            picks.append(i)
            K -= weights[i-1]

    picks.sort()
    picks = [x-1 for x in picks] # change to 0-index

    if return_all:
        max_val = table[n_items,capacity]
        return picks,max_val
    return picks

def check_inputs(values,weights,n_items,capacity):
    # check variable type
    assert(isinstance(values,list))
    assert(isinstance(weights,list))
    assert(isinstance(n_items,int))
    assert(isinstance(capacity,int))
    # check value type
    assert(all(isinstance(val,int) or isinstance(val,float) for val in values))
    assert(all(isinstance(val,int) for val in weights))
    # check validity of value
    assert(all(val >= 0 for val in weights))
    assert(n_items > 0)
    assert(capacity > 0)

def test_knapsack_dp():
    values = [2,3,4]
    weights = [1,2,3]
    n_items = 3
    capacity = 3
    picks = knapsack_dp(values,weights,n_items,capacity)
    print (picks)



osolver = pywrapknapsack_solver.KnapsackSolver(
    # pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
    pywrapknapsack_solver.KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
    'test')

def knapsack_ortools(values, weights, items, capacity ):
    scale = 1000
    values = np.array(values)
    weights = np.array(weights)
    values = (values * scale).astype(np.int64)
    weights = (weights).astype(np.int64)
    capacity = capacity

    osolver.Init(values.tolist(), [weights.tolist()], [capacity])
    computed_value = osolver.Solve()
    packed_items = [x for x in range(0, len(weights))
                    if osolver.BestSolutionContains(x)]

    return packed_items

def my_generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15, method='knapsack'):
    """Generate keyshot-based video summary i.e. a binary vector.
    Args:
    ---------------------------------------------
    - ypred: predicted importance scores.
    - cps: change points, 2D matrix, each row contains a segment.
    - n_frames: original number of frames.
    - nfps: number of frames per segment.
    - positions: positions of subsampled frames in the original video.
    - proportion: length of video summary (compared to original video length).
    - method: defines how shots are selected, ['knapsack', 'rank'].
    """
    cps = torch.squeeze(cps).cpu().numpy()
    positions = torch.squeeze(positions).cpu().numpy()
    n_frames = int(n_frames.item())
    nfps = torch.squeeze(nfps).cpu().numpy()
    n_segs = cps.shape[0]
    ypred = ypred.cpu().numpy()

    frame_scores = np.zeros((n_frames), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i == len(ypred):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = ypred[i]

    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx,0]), int(cps[seg_idx,1]+1)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    limits = int(math.floor(n_frames * proportion))

    if method == 'knapsack':
        #picks = knapsack_dp(seg_score, nfps, n_segs, limits)
        picks = knapsack_ortools(seg_score, nfps, n_segs, limits)
    elif method == 'rank':
        order = np.argsort(seg_score)[::-1].tolist()
        picks = []
        total_len = 0
        for i in order:
            if total_len + nfps[i] < limits:
                picks.append(i)
                total_len += nfps[i]
    else:
        raise KeyError("Unknown method {}".format(method))

    summary = np.zeros((1), dtype=np.float32) # this element should be deleted
    for seg_idx in range(n_segs):
        nf = nfps[seg_idx]
        if seg_idx in picks:
            tmp = np.ones((nf), dtype=np.float32)
        else:
            tmp = np.zeros((nf), dtype=np.float32)
        summary = np.concatenate((summary, tmp))

    summary = np.delete(summary, 0) # delete the first element
    return summary


def my_evaluate_summary(machine_summary, user_summary, eval_metric='avg'):
    """Compare machine summary with user summary (keyshot-based).
    Args:
    --------------------------------
    machine_summary and user_summary should be binary vectors of ndarray type.
    eval_metric = {'avg', 'max'}
    'avg' averages results of comparing multiple human summaries.
    'max' takes the maximum (best) out of multiple comparisons.
    """
    machine_summary = machine_summary.astype(np.float32)
    user_summary = user_summary.astype(np.float32)
    n_users,n_frames = user_summary.shape

    # binarization
    machine_summary[machine_summary > 0] = 1
    user_summary[user_summary > 0] = 1

    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary)))
        machine_summary = np.concatenate([machine_summary, zero_padding])

    f_scores = []
    prec_arr = []
    rec_arr = []

    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx,:]
        overlap_duration = (machine_summary * gt_summary).sum()
        precision = overlap_duration / (machine_summary.sum() + 1e-8)
        recall = overlap_duration / (gt_summary.sum() + 1e-8)
        if precision == 0 and recall == 0:
            f_score = 0.
        else:
            f_score = (2 * precision * recall) / (precision + recall)
        f_scores.append(f_score)
        prec_arr.append(precision)
        rec_arr.append(recall)

    if eval_metric == 'avg':
        final_f_score = np.mean(f_scores)
        final_prec = np.mean(prec_arr)
        final_rec = np.mean(rec_arr)
    elif eval_metric == 'max':
        final_f_score = np.max(f_scores)
        max_idx = np.argmax(f_scores)
        final_prec = prec_arr[max_idx]
        final_rec = rec_arr[max_idx]

    return final_f_score, final_prec, final_rec, f_scores



"""
여기서부터 보시면 됩니다.
"""
def read_from_split(dataset_type, split):
    train_test_json = f"/data/project/rw/video_summarization/PGL-SUM/data/splits/{dataset_type}_splits.json"
    with open(train_test_json, 'r') as fd:
        json_load = json.load(fd)
        video_list = json_load[split]["test_keys"]
    return video_list

def read_from_dataset(dataset_type, video_name):
    #! TODO 
    dataset_path = f"/data/project/rw/video_summarization/PGL-SUM/data/eccv16_dataset_{dataset_type}_google_pool5.h5"
    hdf = h5py.File(dataset_path, 'r')
    gtscore = np.array(hdf[video_name + '/gtscore'])
    user_summary = np.array(hdf[video_name + '/user_summary'])
    num_frames = np.array(hdf[video_name + '/n_frames'])
    picks = torch.tensor(np.array(hdf[video_name + '/picks']), dtype=torch.int64)
    nfps = torch.tensor(np.array(hdf[video_name + '/n_frame_per_seg']), dtype=torch.int64)
    cps = torch.tensor(np.array(hdf[video_name + '/change_points']), dtype=torch.int64)
    hdf.close()
    return gtscore, user_summary, (cps, num_frames, nfps, picks)

def read_from_log(path):
    logits = torch.load(path)
    return logits

def main():
    try:
        dataset_type = sys.argv[1]
        model_path = sys.argv[2]
    except:
        print("<Usage>\n python src/check_summary.py summe 4 logs/train/multiruns/TSt123lr5e4/2022-09-01_09-33-05_summe4_TSt123lr5e4/0/vis")
        exit()
    print("Evaluate from pretrained logits")
    print(f"From {model_path}")
    print(f"Dataset: {dataset_type}")
    print("------------------------------------")
    # python check_summary.py summe 0 /data/project/rw/video_summarization/PGL-SUM/Summaries/PGL-SUM/0902_seed42/SumMe

    mean_score_list = []
    max_score_list = []
    mean_split_score = []
    max_split_score = []

    for split_id in range(5):
        mean_score_list = []
        print('dataset_type', dataset_type)
        max_score_list = []
        exit()
        # log_path = os.path.join(model_path, 'logs' ,f'{dataset_type}_{split_id}')
        print('log_path', log_path)
        logit_file = read_from_log(f"{log_path}/test_logits.pt")
        video_list = read_from_split(dataset_type, split_id)
        for video_name in video_list:
            logits = logit_file[video_name].squeeze()
            gtscore, user_summary, forsummary = read_from_dataset(dataset_type, video_name)
            machine_summary = my_generate_summary(logits, forsummary[0], forsummary[1], forsummary[2], forsummary[3])

            machine_summary_path = os.path.join(log_path, f"gensum_{video_name}.npy")
            fscores_path = os.path.join(log_path, f"fscores_{video_name}.npy")
            fm, _, _, fscores = my_evaluate_summary(machine_summary, user_summary, 'avg')
            # fm, _, _, fscores = evaluate_summary(machine_summary, user_summary, 'max')
            np.save(machine_summary_path, machine_summary)
            np.save(fscores_path, np.array(fscores))


            print_fscores = [round(f1,2) for f1 in fscores]
            # print(f"Video Name: {video_name}")
            # print("All scores: ", print_fscores, "\n")
            # print("mean score", fm)
            # print("max score", max(print_fscores))
            # print("------------------------------------")

            mean_score_list.append(fm)
            max_score_list.append(max(fscores))
        mean_split_score.append(sum(mean_score_list) / len(mean_score_list))
        max_split_score.append(sum(max_score_list) / len(max_score_list))

    for i in range(5):
        print(f"[Split{i}\t Mean eval result: ", mean_split_score[i])
        print(f"[Split{i}\t Max eval result: ", max_split_score[i])
        print()
            
        # print("Mean eval result: ", sum(mean_score_list) / len(mean_score_list))
        # print("Max eval result: ", sum(max_score_list) / len(max_score_list))


if __name__ == "__main__":
    main()