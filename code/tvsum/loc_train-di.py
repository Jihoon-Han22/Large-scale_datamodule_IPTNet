import os
import pickle
import glob
import numpy as np
from itertools import groupby
from operator import itemgetter

import torch.backends.cudnn
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from ortools.algorithms.pywrapknapsack_solver import KnapsackSolver

from loc_nets import LocNets
from helper import Average
import warnings

warnings.filterwarnings("ignore")

def init_loc(args):

    train_loader, num_words, num_chars, word_vectors, train_samples = get_data_loader(args)

    args.num_words = num_words
    args.num_chars = num_chars

    model = LocNets(
        args=args,
        word_vectors=word_vectors
    )
    model_tech = LocNets(
        args=args,
        word_vectors=word_vectors
    )

    model = model.to(args.device)
    model_tech = model_tech.to(args.device)

    return model, model_tech


def get_data_loader(args):

    filename = os.path.join(args.data_dir, 'loc_data.pkl')

    with open(filename, mode='rb') as handle:
        data = pickle.load(handle)

    train_dataset = data['train_set']
    num_train_samples = len(train_dataset)
    num_words = len(data['word_dict'])
    num_chars = len(data['char_dict'])
    word_vectors = data['word_vector']

    visual_features = load_video_features(
        os.path.join('datasets', 'features'),
        max_pos_len=args.max_pos_len
    )

    train_set = Dataset(dataset=train_dataset, video_features=visual_features)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    return train_loader, num_words, num_chars, word_vectors, num_train_samples


def load_video_features(root, max_pos_len):
    video_features = dict()
    filenames = glob.glob(os.path.join(root, "*.npy"))
    for filename in filenames:
        video_id = filename.split("/")[-1].split(".")[0]
        feature = np.load(filename)
        if max_pos_len is None:
            video_features[video_id] = feature
        else:
            new_feature = visual_feature_sampling(feature, max_num_clips=max_pos_len)
            video_features[video_id] = new_feature
    return video_features


def visual_feature_sampling(visual_feature, max_num_clips):
    num_clips = visual_feature.shape[0]
    if max_num_clips is None or num_clips <= max_num_clips:
        return visual_feature
    idxs = np.arange(0, max_num_clips + 1, 1.0) / max_num_clips * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_visual_feature = []
    for i in range(max_num_clips):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_visual_feature.append(np.mean(visual_feature[s_idx:e_idx], axis=0))
        else:
            new_visual_feature.append(visual_feature[s_idx])
    new_visual_feature = np.asarray(new_visual_feature)
    return new_visual_feature


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_features):
        super(Dataset, self).__init__()
        self.dataset = dataset
        self.video_features = video_features

    def __getitem__(self, index):
        record = self.dataset[index]
        video_feature = self.video_features[record['vid']]
        start_time, end_time = float(record['s_time']), float(record['e_time'])
        start_index, end_index = int(record['s_ind']), int(record['e_ind'])
        duration = float(record['duration'])
        num_units = int(record['num_units'])
        word_ids, char_ids = record['word_ids'], record['char_ids']
        return video_feature, word_ids, char_ids, start_time, end_time, start_index, end_index, num_units, duration

    def __len__(self):
        return len(self.dataset)


def collate_fn(data):

    video_features, word_ids, char_ids, s_times, e_times, s_inds, e_inds, num_units, durations = zip(*data)

    word_ids, _ = pad_sequences(word_ids)
    word_ids = np.asarray(word_ids, dtype=np.int32)
    char_ids, _ = pad_char_sequences(char_ids)
    char_ids = np.asarray(char_ids, dtype=np.int32)
    video_features, _ = pad_visual_sequence(video_features)
    video_features = np.asarray(video_features, dtype=np.float32)
    bsz, max_len, _ = video_features.shape

    s_times = np.asarray(s_times, dtype=np.float32)
    e_times = np.asarray(e_times, dtype=np.float32)
    s_labels = np.asarray(s_inds, dtype=np.int64)
    e_labels = np.asarray(e_inds, dtype=np.int64)

    highlight_labels = np.zeros(shape=[bsz, max_len], dtype=np.int32)
    extend = 0.1

    for idx in range(bsz):
        st, et = s_inds[idx], e_inds[idx]
        extend_len = round(extend * float(et - st + 1))
        if extend_len > 0:
            st_ = max(0, st - extend_len)
            et_ = min(et + extend_len, max_len - 1)
            highlight_labels[idx][st_:(et_ + 1)] = 1
        else:
            highlight_labels[idx][st:(et + 1)] = 1

    num_units = np.asarray(num_units, dtype=np.int32)

    video_features = torch.tensor(video_features, dtype=torch.float32)
    word_ids = torch.tensor(word_ids, dtype=torch.int64)
    char_ids = torch.tensor(char_ids, dtype=torch.int64)
    s_times = torch.tensor(s_times, dtype=torch.float32)
    e_times = torch.tensor(e_times, dtype=torch.float32)
    s_labels = torch.tensor(s_labels, dtype=torch.int64)
    e_labels = torch.tensor(e_labels, dtype=torch.int64)
    highlight_labels = torch.tensor(highlight_labels, dtype=torch.int64)
    num_units = torch.tensor(num_units, dtype=torch.int64)
    durations = torch.tensor(durations, dtype=torch.float32)

    return (video_features, word_ids, char_ids, s_times, e_times, s_labels, e_labels, highlight_labels, num_units,
            durations)


def pad_sequences(sequences, pad_tok=None, max_length=None):
    if pad_tok is None:
        pad_tok = 0
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length


def pad_char_sequences(sequences, words_max_length=None, char_max_length=None):
    sequence_padded, sequence_length = [], []
    if words_max_length is None:
        words_max_length = max(map(lambda x: len(x), sequences))
    if char_max_length is None:
        char_max_length = max([max(map(lambda x: len(x), seq)) for seq in sequences])
    for seq in sequences:
        sp, sl = pad_sequences(seq, max_length=char_max_length)
        sequence_padded.append(sp)
        sequence_length.append(sl)
    sequence_padded, _ = pad_sequences(sequence_padded, pad_tok=[0] * char_max_length, max_length=words_max_length)
    sequence_length, _ = pad_sequences(sequence_length, max_length=words_max_length)
    return sequence_padded, sequence_length


def pad_visual_sequence(sequences, max_length=None):
    if max_length is None:
        max_length = max([vfeat.shape[0] for vfeat in sequences])
    feature_dim = sequences[0].shape[1]
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        add_length = max_length - seq.shape[0]
        sequence_length.append(seq.shape[0])
        if add_length > 0:
            add_feature = np.zeros(shape=[add_length, feature_dim], dtype=np.float32)
            seq_ = np.concatenate([seq, add_feature], axis=0)
        else:
            seq_ = seq
        sequence_padded.append(seq_)
    return sequence_padded, sequence_length


def train_loc(
        args,
        model_sum,
        model_sum_tech,
        optimizer,
        model_loc,
        model_loc_tech,
        coll_tr,
        set_epochs,
        train_loader_sum,
        val_loader_sum,
        max_val_fscore,
        save_path
):

    train_loader, num_words, num_chars, word_vectors, train_samples = get_data_loader(args)


    if not coll_tr:

        for epoch in range(set_epochs):

            step_counter = 0
            model_loc.train()

            for video_features, word_ids, char_ids, _, _, s_labels, e_labels, hl_labels, num_units, _ in train_loader:

                step_counter += 1

                video_features = video_features.to(args.device)
                num_units = num_units.to(args.device)
                word_ids = word_ids.to(args.device)
                char_ids = char_ids.to(args.device)
                s_labels = s_labels.to(args.device)
                e_labels = e_labels.to(args.device)
                hl_labels = hl_labels.to(args.device)

                query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(args.device)
                video_mask = convert_length_to_mask(num_units).to(args.device)

                hl_score, start_logits, end_logits = model_loc(
                    word_ids,
                    char_ids,
                    video_features,
                    video_mask,
                    query_mask
                )

                hl_loss = model_loc.compute_highlight_loss(hl_score, hl_labels, video_mask)
                entropy_loss = model_loc.compute_loss(start_logits, end_logits, s_labels, e_labels)
                loc_loss = args.loc_lambda * hl_loss + entropy_loss

                optimizer.zero_grad()
                loc_loss.backward()
                nn.utils.clip_grad_norm_(model_loc.parameters(), args.clip_norm)
                optimizer.step()


                for tech_param, param in zip(model_loc_tech.parameters(), model_loc.parameters()):
                    tech_param.data.mul_(args.ema).add_(1 - args.ema, param.data)

                if step_counter % args.period == 0:
                    print(f'loc epoch: {epoch}, step: {step_counter}, entropy loss: {round(entropy_loss.item(), 4)}, '
                          f'hl loss: {round(hl_loss.item(), 4)}')
                if step_counter >= args.total_step:
                    break

    else:

        for epoch in range(set_epochs):

            step_counter = 0
            model_loc.train()

            for video_features, word_ids, char_ids, _, _, s_labels, e_labels, hl_labels, num_units, _ in train_loader:

                step_counter += 1

                video_features = video_features.to(args.device)
                num_units = num_units.to(args.device)
                word_ids = word_ids.to(args.device)
                char_ids = char_ids.to(args.device)
                s_labels = s_labels.to(args.device)
                e_labels = e_labels.to(args.device)
                hl_labels = hl_labels.to(args.device)

                query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(args.device)
                video_mask = convert_length_to_mask(num_units).to(args.device)

                hl_score, start_logits, end_logits = model_loc(
                    word_ids,
                    char_ids,
                    video_features,
                    video_mask,
                    query_mask
                )
                hl_score_tech, start_logits_tech, end_logits_tech = model_loc_tech(
                    word_ids,
                    char_ids,
                    video_features,
                    video_mask,
                    query_mask
                )

                score_pred, reg_pred, prop_pred = model_sum(video_features)

                prop_hl_loss = F.kl_div(prop_pred[0].softmax(dim=-1).log(), hl_score.softmax(dim=-1), reduction='sum')
                for scale in range(1, len(args.scales)):
                    prop_hl_loss = prop_hl_loss + F.kl_div(prop_pred[scale].softmax(dim=-1).log(), hl_score.softmax(dim=-1), reduction='sum')

                prop_hl_loss_tech = F.kl_div(prop_pred[0].softmax(dim=-1).log(), hl_score_tech.softmax(dim=-1), reduction='sum')
                for scale in range(1, len(args.scales)):
                    prop_hl_loss_tech = prop_hl_loss_tech + F.kl_div(prop_pred[scale].softmax(dim=-1).log(), hl_score_tech.softmax(dim=-1), reduction='sum')


                hl_score_loss = F.kl_div(hl_score.softmax(dim=-1).log(), hl_score_tech.softmax(dim=-1), reduction='sum')

                hl_loss = model_loc.compute_highlight_loss(hl_score, hl_labels, video_mask)
                entropy_loss = model_loc.compute_loss(start_logits, end_logits, s_labels, e_labels)
                loc_loss = args.loc_lambda * hl_loss + entropy_loss

                loss1 = prop_hl_loss + prop_hl_loss_tech
                loss2 = hl_score_loss + loc_loss

                opti_loss = loss1 + loss2
                optimizer.zero_grad()
                opti_loss.backward()
                nn.utils.clip_grad_norm_(model_loc.parameters(), args.clip_norm)
                optimizer.step()


                for tech_param, param in zip(model_loc_tech.parameters(), model_loc.parameters()):
                    tech_param.data.mul_(args.ema).add_(1 - args.ema, param.data)

                for tech_param, param in zip(model_sum_tech.parameters(), model_sum.parameters()):
                    tech_param.data.mul_(args.ema).add_(1 - args.ema, param.data)

                if step_counter % args.eval_period == 0:
                    val_fscore, _, _, _, _ = sum_eval(model_sum, val_loader_sum, args.nms_thresh, args.device, args=args)
                    val_fscore_tech, _, _, _, _ = sum_eval(model_sum_tech, val_loader_sum, args.nms_thresh, args.device, args=args)

                    if step_counter % args.period == 0:
                        print(f'loc epoch: {epoch}, step: {step_counter}')
                        print(f'entropy loss: {round(entropy_loss.item(), 4)}, hl loss: {round(hl_loss.item(), 4)}')
                        print(f'max val fscore: {round(max_val_fscore, 4)}')

                    if max_val_fscore < val_fscore:
                        max_val_fscore = val_fscore
                        torch.save(model_sum.state_dict(), str(save_path))

                    if max_val_fscore < val_fscore_tech:
                        max_val_fscore = val_fscore_tech
                        torch.save(model_sum_tech.state_dict(), str(save_path))

                if step_counter >= args.total_step:
                    break


            for _, seq, gtscore, change_points, n_frames, nfps, picks, user_summary, sum_ratio in train_loader_sum:

                step_counter += 1

                keyshot_summ = get_keyshot_summ(gtscore, change_points, n_frames, nfps, picks)
                summary = keyshot_summ[picks]
                # summary = downsample_summ(keyshot_summ)

                if not summary.any():
                    continue

                target_boundbox = seq_to_boundbox(summary)
                target_boundbox = lr_to_cw(target_boundbox)

                anchors = get_anchors(summary.size, args.scales)
                label_sum, reg_label = pos_sum(anchors, target_boundbox, args.pos_iou_thresh)

                num_pos = label_sum.sum()

                label_sum_neg, _ = pos_sum(anchors, target_boundbox, args.neg_iou_thresh)
                label_sum_neg = neg_sum(label_sum_neg, int(args.neg_sample_ratio * num_pos))

                label_sum_incom, _ = pos_sum(anchors, target_boundbox, args.incomplete_iou_thresh)
                label_sum_incom[label_sum_neg != 1] = 1
                label_sum_incom = neg_sum(label_sum_incom, int(args.incomplete_sample_ratio * num_pos))

                label_sum[label_sum_neg == -1] = -1
                label_sum[label_sum_incom == -1] = -1

                label_sum = torch.tensor(label_sum, dtype=torch.float32).to(args.device)
                reg_label = torch.tensor(reg_label, dtype=torch.float32).to(args.device)

                video_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)

                score_pred, reg_pred, prop_pred = model_sum(video_seq)
                score_pred_tech, reg_pred_tech, prop_pred_tech = model_sum_tech(video_seq)

                score_loss = sum_score_loss(score_pred, label_sum)
                reg_loss = sum_reg_loss(reg_pred, reg_label, label_sum)

                sum_loss = score_loss + reg_loss

                hl_score = model_loc.with_sum_data(video_seq)
                _ = model_loc_tech.with_sum_data(video_seq)

                prop_hl_loss = F.kl_div(prop_pred[0].softmax(dim=-1).log(), hl_score.softmax(dim=-1), reduction='sum')
                for scale in range(1, len(args.scales)):
                    prop_hl_loss = prop_hl_loss + F.kl_div(prop_pred[scale].softmax(dim=-1).log(), hl_score.softmax(dim=-1), reduction='sum')

                prop_hl_loss_tech = F.kl_div(prop_pred_tech[0].softmax(dim=-1).log(), hl_score.softmax(dim=-1), reduction='sum')
                for scale in range(1, len(args.scales)):
                    prop_hl_loss_tech = prop_hl_loss_tech + F.kl_div(prop_pred_tech[scale].softmax(dim=-1).log(), hl_score.softmax(dim=-1), reduction='sum')

                pred_score_loss = F.kl_div(score_pred[:, 0].softmax(dim=-1).log(), score_pred_tech[:, 0].softmax(dim=-1), reduction='sum')
                for scale in range(1, len(args.scales)):
                    pred_score_loss = pred_score_loss + F.kl_div(score_pred[:, scale].softmax(dim=-1).log(), score_pred_tech[:, scale].softmax(dim=-1), reduction='sum')

                loss1 = sum_loss + pred_score_loss
                loss2 = prop_hl_loss + prop_hl_loss_tech

                opti_loss = loss1 + loss2
                optimizer.zero_grad()
                opti_loss.backward()
                optimizer.step()


                for tech_param, param in zip(model_loc_tech.parameters(), model_loc.parameters()):
                    tech_param.data.mul_(args.ema).add_(1 - args.ema, param.data)

                for tech_param, param in zip(model_sum_tech.parameters(), model_sum.parameters()):
                    tech_param.data.mul_(args.ema).add_(1 - args.ema, param.data)


                if step_counter % args.eval_period == 0:
                    val_fscore, _, _, _ , _= sum_eval(model_sum, val_loader_sum, args.nms_thresh, args.device, args=args)
                    val_fscore_tech, _, _, _, _ = sum_eval(model_sum_tech, val_loader_sum, args.nms_thresh, args.device, args=args)

                    print('val fscore: %.4f, max val fscore: %.4f' % (val_fscore, max_val_fscore))
                    print('val fscore tech: %.4f' % (val_fscore_tech))

                    if max_val_fscore < val_fscore:
                        max_val_fscore = val_fscore
                        torch.save(model_sum.state_dict(), str(save_path))

                    if max_val_fscore < val_fscore_tech:
                        max_val_fscore = val_fscore_tech
                        torch.save(model_sum_tech.state_dict(), str(save_path))

    return model_sum, model_sum_tech, optimizer, model_loc, model_loc_tech, max_val_fscore


def convert_length_to_mask(lengths):
    max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(lengths.size()[0], max_len) < lengths.unsqueeze(1)
    mask = mask.float()
    return mask


def sum_eval(model, val_loader, nms_thresh, device, args=None):
    model.eval()
    stats = Average('fscore')
    logits_dict = {}

    with torch.no_grad():
        # for test_key, seq, _, cps, n_frames, nfps, picks, user_summary in val_loader:
        for test_key, seq, gtscore, cps, n_frames, nfps, picks, user_summary, sum_ratio in val_loader:
            """"""
            loss, score_loss, reg_loss, cen_loss = 0., 0., 0., 0.
            logits = []
            tmp_fscore = []

            # for calc val loss
            for i in range(len(user_summary)):
            
                keyshot_summ = get_keyshot_summ(gtscore, cps, n_frames, nfps, picks, proportion=sum_ratio[i])
                summary = keyshot_summ[picks]

                if not summary.any():
                    continue

                target_boundbox = seq_to_boundbox(summary)
                target_boundbox = lr_to_cw(target_boundbox)

                anchors = get_anchors(summary.size, args.scales)
                label_sum, reg_label = pos_sum(anchors, target_boundbox, args.pos_iou_thresh)

                num_pos = label_sum.sum()

                label_sum_neg, _ = pos_sum(anchors, target_boundbox, args.neg_iou_thresh)
                label_sum_neg = neg_sum(label_sum_neg, int(args.neg_sample_ratio * num_pos))

                label_sum_incom, _ = pos_sum(anchors, target_boundbox, args.incomplete_iou_thresh)
                label_sum_incom[label_sum_neg != 1] = 1
                label_sum_incom = neg_sum(label_sum_incom, int(args.incomplete_sample_ratio * num_pos))

                label_sum[label_sum_neg == -1] = -1
                label_sum[label_sum_incom == -1] = -1

                label_sum = torch.tensor(label_sum, dtype=torch.float32).to(device)
                reg_label = torch.tensor(reg_label, dtype=torch.float32).to(device)
                """"""

                seq_len = len(seq)
                seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)

                score_pred, pred_boundbox, val_loss, score_loss, reg_loss = model.predict(seq_torch,label_sum, reg_label)

                pred_boundbox = np.clip(pred_boundbox, 0, seq_len).round().astype(np.int32)

                score_pred, pred_boundbox = nms(
                    score_pred,
                    pred_boundbox,
                    nms_thresh
                )

                pred_summ, t_logits = boundbox_to_summary(
                    seq_len,
                    score_pred,
                    pred_boundbox,
                    cps,
                    n_frames,
                    nfps,
                    picks,
                    sum_ratio[i]
                )
                eval_metric = 'avg' if 'tvsum' in test_key else 'max'
                tmp_fscore.append(get_summ_f1score(pred_summ, user_summary[i]))
                logits.append(t_logits)

            if eval_metric == 'avg':
                final_f1 = np.mean(tmp_fscore)
            elif eval_metric == 'max':
                final_f1 = np.max(tmp_fscore)
            else:
                final_f1 = 0.
                print('Eval metric error.')
            logits_dict[test_key] = logits

            stats.update(fscore=final_f1)

    return stats.fscore, val_loss, score_loss, reg_loss, logits_dict


def nms(scores, boundbox, thresh):

    valid_idx = boundbox[:, 0] < boundbox[:, 1]
    scores = scores[valid_idx]
    boundbox = boundbox[valid_idx]

    arg_desc = scores.argsort()[::-1]

    scores_remain = scores[arg_desc]
    boundbox_remain = boundbox[arg_desc]

    keep_boundbox = []
    keep_scores = []

    while boundbox_remain.size > 0:
        bbox = boundbox_remain[0]
        score = scores_remain[0]
        keep_boundbox.append(bbox)
        keep_scores.append(score)

        iou = iou_lr(boundbox_remain, np.expand_dims(bbox, axis=0))

        keep_indices = (iou < thresh)
        boundbox_remain = boundbox_remain[keep_indices]
        scores_remain = scores_remain[keep_indices]

    keep_boundbox = np.asarray(keep_boundbox, dtype=boundbox.dtype)
    keep_scores = np.asarray(keep_scores, dtype=scores.dtype)

    return keep_scores, keep_boundbox


def iou_lr(anchor_boundbox, target_boundbox):

    anchor_left, anchor_right = anchor_boundbox[:, 0], anchor_boundbox[:, 1]
    target_left, target_right = target_boundbox[:, 0], target_boundbox[:, 1]

    inter_left = np.maximum(anchor_left, target_left)
    inter_right = np.minimum(anchor_right, target_right)
    union_left = np.minimum(anchor_left, target_left)
    union_right = np.maximum(anchor_right, target_right)

    intersect = inter_right - inter_left
    intersect[intersect < 0] = 0
    union = union_right - union_left
    union[union <= 0] = 1e-6

    iou = intersect / union
    return iou


def boundbox_to_summary(
        seq_len,
        score_pred,
        pred_boundbox,
        change_points,
        n_frames,
        nfps,
        picks,
        sum_ratio
):
    score = np.zeros(seq_len, dtype=np.float32)

    for idx in range(len(pred_boundbox)):
        lo, hi = pred_boundbox[idx, 0], pred_boundbox[idx, 1]
        score[lo:hi] = np.maximum(score[lo:hi], [score_pred[idx]])

    pred_summ = get_keyshot_summ(score, change_points, n_frames, nfps, picks, proportion=sum_ratio)
    return pred_summ, score


def get_keyshot_summ(
        pred,
        cps,
        n_frames,
        nfps,
        picks,
        proportion=0.15
):

    assert pred.shape == picks.shape
    picks = np.asarray(picks, dtype=np.int32)

    frame_scores = np.zeros(n_frames, dtype=np.float32)
    for i in range(len(picks)):
        pos_lo = picks[i]
        pos_hi = picks[i + 1] if i + 1 < len(picks) else n_frames
        frame_scores[pos_lo:pos_hi] = pred[i]

    seg_scores = np.zeros(len(cps), dtype=np.int32)
    for seg_idx, (first, last) in enumerate(cps):
        scores = frame_scores[first:last + 1]
        seg_scores[seg_idx] = int(1000 * scores.mean())

    limits = int(n_frames * proportion)
    packed = knapsack(seg_scores, nfps, limits)

    summary = np.zeros(n_frames, dtype=np.bool)
    for seg_idx in packed:
        first, last = cps[seg_idx]
        summary[first:last + 1] = True

    return summary


def knapsack(
        values,
        weights,
        capacity
):

    knapsack_solver = KnapsackSolver(KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER, 'test')

    values = list(values)
    weights = list(weights)
    capacity = int(capacity)

    knapsack_solver.Init(values, [weights], [capacity])
    knapsack_solver.Solve()
    packed_items = [x for x in range(0, len(weights)) if knapsack_solver.BestSolutionContains(x)]

    return packed_items


def get_summ_f1score(
        pred_summ,
        test_summ
):

    pred_summ = np.asarray(pred_summ, dtype=np.bool)
    test_summ = np.asarray(test_summ, dtype=np.bool)
    n_frames = test_summ.shape[0]

    if pred_summ.size > n_frames:
        pred_summ = pred_summ[:n_frames]
    elif pred_summ.size < n_frames:
        pred_summ = np.pad(pred_summ, (0, n_frames - pred_summ.size), mode='constant')

    f1 = f1_score(pred_summ, test_summ)

    # if eval_metric == 'avg':
    #     final_f1 = np.mean(f1s)
    # elif eval_metric == 'max':
    #     final_f1 = np.max(f1s)
    # else:
    #     final_f1 = 0.
    #     print('Eval metric error.')

    return float(f1)


def f1_score(pred, test):

    assert pred.shape == test.shape

    pred = np.asarray(pred, dtype=np.bool)
    test = np.asarray(test, dtype=np.bool)
    overlap = (pred & test).sum()
    if overlap == 0:
        return 0.0
    precision = overlap / pred.sum()
    recall = overlap / test.sum()
    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


def downsample_summ(summ):

    return summ[::15]


def get_reg_label(summary):

    seq_len, = summary.shape

    boundbox = seq_to_boundbox(summary)
    offsets = boundbox_to_offset(boundbox, seq_len)

    return offsets


def seq_to_boundbox(sequence):
    sequence = np.asarray(sequence, dtype=np.bool)
    selected_indices, = np.where(sequence == 1)

    boundbox_lr = []
    for k, g in groupby(enumerate(selected_indices), lambda x: x[0] - x[1]):
        segment = list(map(itemgetter(1), g))
        start_frame, end_frame = segment[0], segment[-1] + 1
        boundbox_lr.append([start_frame, end_frame])

    boundbox_lr = np.asarray(boundbox_lr, dtype=np.int32)
    return boundbox_lr


def pos_sum(anchors, targets, iou_thresh):

    seq_len, num_scales, _ = anchors.shape
    anchors = np.reshape(anchors, (seq_len * num_scales, 2))

    reg_label = np.zeros((seq_len * num_scales, 2))
    label_sum = np.zeros(seq_len * num_scales, dtype=np.int32)

    for target in targets:
        target = np.tile(target, (seq_len * num_scales, 1))
        iou = iou_cw(anchors, target)
        pos_idx = np.where(iou > iou_thresh)
        label_sum[pos_idx] = 1
        reg_label[pos_idx] = boundbox_to_offset(target[pos_idx], anchors[pos_idx])

    reg_label = reg_label.reshape((seq_len, num_scales, 2))
    label_sum = label_sum.reshape((seq_len, num_scales))

    return label_sum, reg_label


def neg_sum(label_sum, num_neg):

    seq_len, num_scales = label_sum.shape
    label_sum = label_sum.copy().reshape(-1)
    label_sum[label_sum < 0] = 0

    neg_idx, = np.where(label_sum == 0)
    np.random.shuffle(neg_idx)
    neg_idx = neg_idx[:num_neg]

    label_sum[neg_idx] = -1
    label_sum = np.reshape(label_sum, (seq_len, num_scales))
    return label_sum


def iou_cw(anchor_boundbox, target_boundbox):
    anchor_boundbox_lr = cw_to_lr(anchor_boundbox)
    target_boundbox_lr = cw_to_lr(target_boundbox)
    return iou_lr(anchor_boundbox_lr, target_boundbox_lr)


def lr_to_cw(boundbox_lr):

    boundbox_lr = np.asarray(boundbox_lr, dtype=np.float32).reshape((-1, 2))

    center = (boundbox_lr[:, 0] + boundbox_lr[:, 1]) / 2
    width = boundbox_lr[:, 1] - boundbox_lr[:, 0]

    boundbox_cw = np.vstack((center, width)).T
    return boundbox_cw


def cw_to_lr(boundbox_cw):

    boundbox_cw = np.asarray(boundbox_cw, dtype=np.float32).reshape((-1, 2))

    left = boundbox_cw[:, 0] - boundbox_cw[:, 1] / 2
    right = boundbox_cw[:, 0] + boundbox_cw[:, 1] / 2

    boundbox_lr = np.vstack((left, right)).T
    return boundbox_lr


def get_anchors(length, scales):

    anchors = np.zeros((length, len(scales), 2), dtype=np.int32)

    for pos in range(length):
        for scale_idx, scale in enumerate(scales):
            anchors[pos][scale_idx] = [pos, scale]

    return anchors


def boundbox_to_offset(boundbox, anchors):

    boundbox_center, boundbox_width = boundbox[:, 0], boundbox[:, 1]
    anchor_center, anchor_width = anchors[:, 0], anchors[:, 1]

    offset_center = (boundbox_center - anchor_center) / anchor_width
    offset_width = np.log(boundbox_width / anchor_width)

    offset = np.vstack((offset_center, offset_width)).T
    return offset


def get_cen_label(
        summary,
        offset,
        eps=1e-8
):

    summary = np.asarray(summary, dtype=np.bool)
    cen_label = np.zeros(summary.shape, dtype=np.float32)

    offset_left, offset_right = offset[summary, 0], offset[summary, 1]

    cen_label[summary] = np.minimum(offset_left, offset_right) / (np.maximum(offset_left, offset_right) + eps)

    return cen_label


def sum_score_loss(score_pred, label_sum):

    score_pred = score_pred.view(-1)
    label_sum = label_sum.view(-1)

    pos_idx = label_sum.eq(1).nonzero().squeeze(-1)
    pred_pos = score_pred[pos_idx].unsqueeze(-1)
    pred_pos = torch.cat([1 - pred_pos, pred_pos], dim=-1)
    gt_pos = torch.ones(pred_pos.shape[0], dtype=torch.long, device=score_pred.device)
    loss_pos = F.nll_loss(pred_pos.log(), gt_pos)

    neg_idx = label_sum.eq(-1).nonzero().squeeze(-1)
    pred_neg = score_pred[neg_idx].unsqueeze(-1)
    pred_neg = torch.cat([1 - pred_neg, pred_neg], dim=-1)
    gt_neg = torch.zeros(pred_neg.shape[0], dtype=torch.long, device=score_pred.device)
    loss_neg = F.nll_loss(pred_neg.log(), gt_neg)

    loss = (loss_pos + loss_neg) * 0.5
    return loss


def focal_loss(
        x,
        y,
        alpha=0.25,
        gamma=2
):

    _, num_classes = x.shape

    t = one_hot_embedding(y, num_classes)

    p_t = x * t + (1 - x) * (1 - t)
    alpha_t = alpha * t + (1 - alpha) * (1 - t)
    fl = -alpha_t * (1 - p_t).pow(gamma) * p_t.log()

    fl = fl.sum()
    return fl


def one_hot_embedding(labels, num_classes):

    eye = torch.eye(num_classes, device=labels.device)
    return eye[labels]


def sum_reg_loss(
        reg_pred,
        reg_label,
        label_sum
):

    pos_idx = label_sum.eq(1).unsqueeze(-1).repeat((1, 1, 2))

    reg_pred = reg_pred[pos_idx]
    reg_label = reg_label[pos_idx]

    loc_loss = F.smooth_l1_loss(reg_pred, reg_label)

    return loc_loss


def iou_offset(
        offset_a,
        offset_b,
        eps: float = 1e-8
):

    left_a, right_a = offset_a[:, 0], offset_a[:, 1]
    left_b, right_b = offset_b[:, 0], offset_b[:, 1]

    length_a = left_a + right_a
    length_b = left_b + right_b

    intersect = torch.min(left_a, left_b) + torch.min(right_a, right_b)
    intersect[intersect < 0] = 0
    union = length_a + length_b - intersect
    union[union <= 0] = eps

    iou = intersect / union
    return iou


def sum_cen_loss(cen_pred, cen_label, target):
    target = target.type(torch.uint8)

    cen_pred = cen_pred[target]
    cen_label = cen_label[target]

    loss = F.binary_cross_entropy(cen_pred, cen_label)
    return loss
