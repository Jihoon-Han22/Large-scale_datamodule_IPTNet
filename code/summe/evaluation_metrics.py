# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
import math
from knapsack_implementation import knapSack

def evaluate_summary(predicted_summary, user_summary, score, gtscore=None, eval_method='avg'):
    """ Compare the predicted summary with the user defined one(s).

    :param ndarray predicted_summary: The generated summary from our model.
    :param ndarray gt_summary: The user defined ground truth summaries (or summary).
    """
    kTs = []
    pSs = []

    # evaluation method for summe and tvsum dataset also applicable on yt8m
    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary
    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G
        # Compute precision, recall, f-score
        precision = sum(overlapped)/sum(S+1e-8)
        recall = sum(overlapped)/sum(G+1e-8)
        # print("max_len", max_len)
        # print("sum of overlapped", sum(overlapped))
        # print("sum of S", sum(S))
        # print("sum of G", sum(G))
        # print("shape of S", np.array(S).shape)
        # print("shape of G", np.array(G).shape)
        # print("precision", precision)
        # print("recall", recall)
        if gtscore is not None:
            gtscore = gtscore.cpu().numpy()
            if gtscore.ndim == 1:
                gtscore = np.expand_dims(gtscore, axis=0)
            
            GS = np.zeros(max_len, dtype=float)
            GS[:gtscore.shape[1]] = gtscore[user]
        
            kTs.append(stats.kendalltau(stats.rankdata(-np.array(GS)), stats.rankdata(-np.array(score)))[0])
            pSs.append(stats.spearmanr(GS, score)[0])

        if precision+recall == 0:
            f_scores.append(0)
        else:
            f_scores.append((2 * precision * recall * 100) / (precision + recall))

    if eval_method == 'max':
        f_score_result = max(f_scores)
    else:
        f_score_result = sum(f_scores)/len(f_scores)
    
    # y_pred2=predicted_summary
    # y_true2=user_summary.mean(axis=0)
    # pS=stats.spearmanr(y_pred2,y_true2)[0]
    # kT=stats.kendalltau(stats.rankdata(-np.array(y_true2)), stats.rankdata(-np.array(y_pred2)))[0]
    
    return f_score_result, np.mean(kTs), np.mean(pSs)

def pick_to_summary(picks, element):
    summary = [0] * element
    for index in picks:
        summary[index] = 1
    return summary

def critical_item(nfps, score, limit):
    profit_per_weight = [[score[i] / nfps[i], i] for i in range(len(nfps))]
    
    # suppose we collect items in the order of profit per weight ratio
    # if the sum of weights exceeds the limit,
    # the first item that exceeds the limit is the critical item
    # return the ratio of the critical item
    
    # Sort items by profit per weight ratio in descending order
    profit_per_weight.sort(reverse=True, key=lambda x: x[0])
    
    total_weight = 0
    
    # Iterate through sorted items
    for ratio, index in profit_per_weight:
        total_weight += nfps[index]
        if total_weight > limit:
            return ratio
    
    return -1  # Return -1 if no critical item is found
            
            

def evaluate_knapsack_opt(score, gtscore, gtsummary, cps, n_frames, nfps, positions):
    n_segs = len(cps)
    n_frames = n_frames[0]
    # if gtscore is numpy
    if isinstance(gtscore, np.ndarray):
        pass
    else:
        gtscore = gtscore.cpu().numpy()
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    frame_gtscores = np.zeros((n_frames), dtype=np.float32)
    gtsummary = gtsummary.squeeze()
    
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
        
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i == len(score):
            frame_scores[pos_left:pos_right] = 0
            frame_gtscores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = score[i]
            frame_gtscores[pos_left:pos_right] = gtscore[i]
            
    seg_score = []
    seg_gtscore = []
    seg_gtsummary = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx][0]), int(cps[seg_idx][1]+1)
        scores = frame_scores[start:end]
        gtscores = frame_gtscores[start:end]
        gtsummaries = gtsummary[start:end-1]
        
        seg_score.append(float(scores.mean()))
        seg_gtscore.append(float(gtscores.mean()))
        seg_gtsummary.append(int(gtsummaries.sum() > 0))
        
    limits = int(math.floor(n_frames * 0.15))
    
    vKP = np.dot(np.array(seg_gtsummary), np.array(seg_gtscore))
    delta_plus = 0
    delta_minus = 0
    psi_p = [0]
    psi_m = [0]
    rho_i = [0]
    deltas = []
    
    WSE = 0
    
    # if any element of nfps is greater than limits, remove the element from nfps, seg_gtscore, seg_gtsummary, and seg_score
    delete_indices = []
    for i in range(len(nfps)):
        if nfps[i] > limits:
            delete_indices.append(i)
            
    nfps = [nfps[i] for i in range(len(nfps)) if i not in delete_indices]
    seg_gtscore = [seg_gtscore[i] for i in range(len(seg_gtscore)) if i not in delete_indices]
    seg_gtsummary = [seg_gtsummary[i] for i in range(len(seg_gtsummary)) if i not in delete_indices]
    seg_score = [seg_score[i] for i in range(len(seg_score)) if i not in delete_indices]
    
    
    for i in range(len(seg_score)):
        if seg_gtsummary[i] == 1:  # 1
            new_nfps = nfps[:i] + nfps[i+1:]
            new_seg_gtscore = seg_gtscore[:i] + seg_gtscore[i+1:]
            pick = knapSack(limits, new_nfps, new_seg_gtscore, len(new_nfps))
            new_summary = pick_to_summary(pick, len(new_nfps))
            vKP_i = np.dot(np.array(new_summary), np.array(new_seg_gtscore))
            if seg_score[i] < seg_gtscore[i]: # 1-
                delta_minus += (seg_score[i] - seg_gtscore[i])
                psi_m.append(vKP_i)
                WSE += -nfps[i] * (seg_score[i] - seg_gtscore[i])
            rho_i.append(critical_item(new_nfps, new_seg_gtscore, limits))
            deltas.append(vKP_i - vKP)
            
        elif seg_gtsummary[i] == 0: # 0
            new_nfps = nfps[:i] + nfps[i+1:]
            new_seg_gtscore = seg_gtscore[:i] + seg_gtscore[i+1:]
            pick = knapSack(limits - nfps[i], new_nfps, new_seg_gtscore, len(new_nfps))
            new_summary = pick_to_summary(pick, len(new_nfps))
            vKP_i = np.dot(np.array(new_summary), np.array(new_seg_gtscore)) + seg_gtscore[i]
            if seg_score[i] > seg_gtscore[i]: # 0+
                delta_plus += seg_score[i] - seg_gtscore[i]
                psi_p.append(vKP_i)
                WSE += nfps[i] * (seg_score[i] - seg_gtscore[i])
            rho_i.append(critical_item(new_nfps, new_seg_gtscore, limits - nfps[i]))
            deltas.append(vKP - vKP_i)
            
    CIS = delta_plus - delta_minus - vKP + max(max(psi_p), max(psi_m))
    
    total = np.sum(nfps)
    count = 0
    cnt = 0
    
    # lower_deltas = []
    # upper_deltas = []
    
    for i in range(len(seg_score)):
        if seg_gtsummary[i] == 1:
            lower_delta = max(deltas[i], nfps[i] * max(rho_i) - seg_gtscore[i])
            # lower_deltas.append(lower_delta)
            if seg_score[i] - seg_gtscore[i] > lower_delta or seg_score[i] > seg_gtscore[i]:
                count += nfps[i]
                cnt += 1
                
        elif seg_gtsummary[i] == 0:
            upper_delta = min(deltas[i], seg_gtscore[i] - nfps[i] * min(rho_i))
            # upper_deltas.append(upper_delta)
            if seg_score[i] - seg_gtscore[i] < upper_delta or seg_score[i] < seg_gtscore[i]:
                count += nfps[i]
                cnt += 1
                
    WIR = count / total
    IR = cnt / len(seg_score)
    
    return WSE, CIS, WIR, IR
    