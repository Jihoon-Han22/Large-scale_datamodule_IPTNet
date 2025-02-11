import os
import math
import pickle
import glob
import numpy as np
from itertools import groupby
from operator import itemgetter
from sklearn.metrics import average_precision_score

import torch.backends.cudnn
import torch.utils.data
import torch.nn as nn
from loc_nets import LocNets
import torch.nn.functional as F
from helper import Average
# from ortools.algorithms.pywrapknapsack_solver import KnapsackSolver
from knapsack_implementation import knapSack
from evaluation_metrics import evaluate_summary, evaluate_knapsack_opt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pdb
from tqdm import tqdm
import copy
from itertools import zip_longest


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
        optimizer1,
        optimizer2,
        model_loc,
        model_loc_tech,
        coll_tr,
        set_epochs,
        train_loader_sum,
        val_loader_sum,
        max_val_fscore,
        save_path,
        writer,
        global_step
):

    train_loader, num_words, num_chars, word_vectors, train_samples = get_data_loader(args)


    if not coll_tr:

        for epoch in range(set_epochs):

            step_counter = 0
            model_loc.train()

            num_batches = int(len(train_loader))
            iterator = iter(train_loader)

            for _ in tqdm(range(num_batches)):
                
                video_features, word_ids, char_ids, _, _, s_labels, e_labels, hl_labels, num_units, _ = next(iterator)

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

                optimizer2.zero_grad()
                loc_loss.backward()
                nn.utils.clip_grad_norm_(model_loc.parameters(), args.clip_norm)
                optimizer2.step()


                for tech_param, param in zip(model_loc_tech.parameters(), model_loc.parameters()):
                    tech_param.data.mul_(args.ema).add_(1 - args.ema, param.data)

                if step_counter % args.period == 0:
                    print('loc epoch: %d, step: %d, entropy loss: %.4f, hl loss: %.4f'
                          % (epoch, step_counter, entropy_loss.item(), hl_loss.item()))
                
                torch.save(model_loc.state_dict(), os.path.join(save_path, 'model_loc.pt'))
                torch.save(model_loc_tech.state_dict(), os.path.join(save_path, 'model_loc_tech.pt'))
                if step_counter >= args.total_step:
                    break

    else:

        for epoch in range(set_epochs):

            step_counter = 0
            model_loc.train()

            num_batches = int(len(train_loader))
            iterator = iter(train_loader)
        
            for _ in tqdm(range(num_batches)):
                video_features, word_ids, char_ids, _, _, s_labels, e_labels, hl_labels, num_units, _ = next(iterator)

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

                score_pred, reg_pred, cen_pred, prop_pred = model_sum(video_features)

                score_prop_loss = F.kl_div(prop_pred.softmax(dim=-1).log(), hl_score.softmax(dim=-1), reduction='sum')

                score_prop_loss_tech = F.kl_div(prop_pred.softmax(dim=-1).log(), hl_score_tech.softmax(dim=-1), reduction='sum')

                _, _, _, _ = model_sum_tech(video_features)


                hl_score_loss = F.kl_div(hl_score.softmax(dim=-1).log(), hl_score_tech.softmax(dim=-1), reduction='sum')

                hl_loss = model_loc.compute_highlight_loss(hl_score, hl_labels, video_mask)
                entropy_loss = model_loc.compute_loss(start_logits, end_logits, s_labels, e_labels)
                loc_loss = args.loc_lambda * hl_loss + entropy_loss

                loss1 = score_prop_loss + score_prop_loss_tech
                loss2 = hl_score_loss + loc_loss

                # optimizer1.zero_grad()
                # loss1.backward(retain_graph=True)
                # optimizer1.step()

                # optimizer2.zero_grad()
                # loss2.backward()
                # nn.utils.clip_grad_norm_(model_loc.parameters(), args.clip_norm)
                # optimizer2.step()

                """ YS"""
                optimizer1.zero_grad()
                loss1.backward(retain_graph=True)
                optimizer2.zero_grad()
                loss2.backward()
                optimizer1.step()
                nn.utils.clip_grad_norm_(model_loc.parameters(), args.clip_norm)
                optimizer2.step()



                for tech_param, param in zip(model_loc_tech.parameters(), model_loc.parameters()):
                    tech_param.data.mul_(args.ema).add_(1 - args.ema, param.data)

                for tech_param, param in zip(model_sum_tech.parameters(), model_sum.parameters()):
                    tech_param.data.mul_(args.ema).add_(1 - args.ema, param.data)

                if step_counter % (args.period * 10) == 0:
                    print('loc epoch: %d, step: %d, entropy loss: %.4f, hl loss: %.4f'
                          % (epoch, step_counter, entropy_loss.item(), hl_loss.item()))

                if step_counter % args.eval_period == 0: # 10마다
                    val_fscore, _, _, _, _, _, map50, map15 = sum_eval(model_sum, val_loader_sum, args.nms_thresh, args.device)
                    val_fscore_tech, _, _, _, _, _, map50, map15 = sum_eval(model_sum_tech, val_loader_sum, args.nms_thresh, args.device)

                    if max_val_fscore < val_fscore:
                        max_val_fscore = val_fscore
                        torch.save(model_sum.state_dict(), os.path.join(save_path, 'model_sum_best.pt'))
                    torch.save(model_sum.state_dict(), os.path.join(save_path, 'model_sum.pt'))

                    if max_val_fscore < val_fscore_tech:
                        max_val_fscore = val_fscore_tech
                        torch.save(model_sum_tech.state_dict(), os.path.join(save_path, 'model_sum_best.pt'))
                    torch.save(model_sum_tech.state_dict(), os.path.join(save_path, 'model_sum_tech.pt'))
                    torch.save(model_loc.state_dict(), os.path.join(save_path, 'model_loc.pt'))
                    torch.save(model_loc_tech.state_dict(), os.path.join(save_path, 'model_loc_tech.pt'))

                    global_step += 1
                    writer.update_loss(val_fscore, global_step, 'val/f1_epoch')
                    writer.update_loss(max_val_fscore, global_step, 'val/f1_best')

                    if step_counter % args.period == 0: # 100마다 찍는거 (그래서 log.txt에 max val fscore에 없던 게 나올 수 있음-아래에서 업뎃하는 과정은 프린트안하기 때문)
                        print('val fscore: %.4f, max val fscore: %.4f' % (val_fscore, max_val_fscore))
                        print('val fscore tech: %.4f' % (val_fscore_tech))
                        print('map 50: %.4f, map 15: %.4f' % (map50, map15))

                if step_counter >= args.total_step:
                    break
        
            with torch.autograd.set_detect_anomaly(True):

                num_batches2 = int(len(train_loader_sum))
                iterator2 = iter(train_loader_sum)

                for _ in tqdm(range(num_batches2)):
                    data = next(iterator2)

                    video_seq = data['features']
                    gtscore = data['gtscore']
                    change_points = data['change_points']
                    n_frames = data['n_frames']
                    nfps = data['n_frame_per_seg']
                    picks = data['picks']
                    user_summary = data['gt_summary']
                    mask = data['mask']
                    mask = mask.to(args.device)

                    step_counter += 1
                    summary = get_keyshot_summ(gtscore, change_points, n_frames, nfps, picks)
                    # summary = keyshot_summ[picks]
                    # summary = downsample_summ(keyshot_summ)

                    if not summary.any():
                        continue

                    video_seq = torch.tensor(video_seq, dtype=torch.float32).to(args.device)

                    reg_label = get_reg_label(summary)
                    cen_label = get_cen_label(summary, reg_label)

                    score_pred, reg_pred, cen_pred, prop_pred = model_sum(video_seq)
                    score_pred_tech, reg_pred_tech, cen_pred_tech, prop_pred_tech = model_sum_tech(video_seq)

                    summary = summary.astype(np.float32)

                    summary = torch.tensor(summary, dtype=torch.float32).to(args.device)
                    reg_label = torch.tensor(reg_label, dtype=torch.float32).to(args.device)
                    cen_label = torch.tensor(cen_label, dtype=torch.float32).to(args.device)

                    score_pred = torch.squeeze(score_pred, 0)
                    reg_pred = torch.squeeze(reg_pred, 0)
                    cen_pred = torch.squeeze(cen_pred, 0)

                    score_loss = sum_score_loss(score_pred, summary, mask)
                    reg_loss = sum_reg_loss(reg_pred, reg_label, summary)
                    cen_loss = sum_cen_loss(cen_pred, cen_label, summary)

                    sum_loss = score_loss + reg_loss + cen_loss

                    hl_score = model_loc.with_sum_data(video_seq)
                    _ = model_loc_tech.with_sum_data(video_seq)

                    hl_score_loss = F.kl_div(prop_pred.softmax(dim=-1).log(), hl_score.softmax(dim=-1), reduction='sum')
                    hl_score_loss_tech = F.kl_div(prop_pred_tech.softmax(dim=-1).log(), hl_score.softmax(dim=-1), reduction='sum')

                    score_pred_loss = F.kl_div(score_pred.softmax(dim=-1).log(), score_pred_tech.softmax(dim=-1), reduction='sum')
                    loss1 = sum_loss + score_pred_loss
                    loss2 = hl_score_loss + hl_score_loss_tech


                    # optimizer1.zero_grad()
                    # loss1.backward(retain_graph=True)
                    # optimizer1.step()

                    # optimizer2.zero_grad()
                    # loss2.backward()
                    # optimizer2.step()
                    """ YS"""
                    optimizer1.zero_grad()
                    loss1.backward(retain_graph=True)
                    optimizer2.zero_grad()
                    loss2.backward()
                    optimizer1.step()
                    nn.utils.clip_grad_norm_(model_loc.parameters(), args.clip_norm)
                    optimizer2.step()

                for tech_param, param in zip(model_loc_tech.parameters(), model_loc.parameters()):
                    tech_param.data.mul_(args.ema).add_(1 - args.ema, param.data)

                for tech_param, param in zip(model_sum_tech.parameters(), model_sum.parameters()):
                    tech_param.data.mul_(args.ema).add_(1 - args.ema, param.data)


                if step_counter % args.eval_period == 0:
                    val_fscore, _, _, _, _ ,_, map50, map15= sum_eval(model_sum, val_loader_sum, args.nms_thresh, args.device)
                    val_fscore_tech, _, _, _, _, _, map50, map15 = sum_eval(model_sum_tech, val_loader_sum, args.nms_thresh, args.device)

                    
                    if max_val_fscore < val_fscore:
                        max_val_fscore = val_fscore
                        torch.save(model_sum.state_dict(), os.path.join(save_path, 'model_sum_best.pt'))

                    if max_val_fscore < val_fscore_tech:
                        max_val_fscore = val_fscore_tech
                        torch.save(model_sum_tech.state_dict(), os.path.join(save_path, 'model_sum_best.pt'))
                    
                    torch.save(model_sum.state_dict(), os.path.join(save_path, 'model_sum.pt'))
                    torch.save(model_sum_tech.state_dict(), os.path.join(save_path, 'model_sum_tech.pt'))
                    torch.save(model_loc.state_dict(), os.path.join(save_path, 'model_loc.pt'))
                    torch.save(model_loc_tech.state_dict(), os.path.join(save_path, 'model_loc_tech.pt'))

                    global_step += 1
                    writer.update_loss(val_fscore, global_step, 'val/f1_epoch')
                    writer.update_loss(max_val_fscore, global_step, 'val/f1_best')
                    
                    print('val fscore: %.4f, max val fscore: %.4f' % (val_fscore, max_val_fscore))
                    print('val fscore tech: %.4f' % (val_fscore_tech))
                    print('map 50: %.4f, map 15: %.4f' % (map50, map15))


    return model_sum, model_sum_tech, optimizer1, optimizer2, model_loc, model_loc_tech, max_val_fscore, writer, global_step


def convert_length_to_mask(lengths):
    max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(lengths.size()[0], max_len) < lengths.unsqueeze(1)
    mask = mask.float()
    return mask


def sum_eval(model, val_loader, nms_thresh, device):
    model.eval()
    stats = Average('fscore')
    
    map50_history = []
    map15_history = []
    sRho_history = []
    kTau_history = []
    WIR_history = []
    WSE_history = []
    IR_history = []
    CIS_history = []
    
    logits_dict = {}

    with torch.no_grad():

        num_batches = int(len(val_loader))
        iterator = iter(val_loader)

        for _ in tqdm(range(num_batches)):
            
            data = next(iterator)
            # _, video_seq, gtscore, change_points, n_frames, nfps, picks, _ = next(iterator)

            test_key = data['video_name']
            seq = data['features']
            gtscore = data['gtscore']
            cps = data['change_points']
            n_frames = data['n_frames']
            nfps = data['n_frame_per_seg']
            picks = data['picks']
            user_summary = data['gt_summary']
            mask = data['mask']
        
            loss, score_loss, reg_loss, cen_loss = 0., 0., 0., 0.
            logits = []
            tmp_fscore = []
            for i in range(len(user_summary)):
                summary = get_keyshot_summ(gtscore, cps, n_frames, nfps, picks)
                # summary = downsample_summ(keyshot_summ)
                # summary = keyshot_summ[picks]
                maski = mask[i].to('cuda')

                seq_len = seq.shape[1]
                seq_torch = seq.to(device)
            
                # for calc val loss
                reg_label = get_reg_label(summary)
                cen_label = get_cen_label(summary, reg_label)
                summary = summary.astype(np.float32)

                summary = torch.tensor(summary, dtype=torch.float32).to('cuda')
                reg_label = torch.tensor(reg_label, dtype=torch.float32).to('cuda')
                cen_label = torch.tensor(cen_label, dtype=torch.float32).to('cuda')
                """"""
            
                score_pred, pred_boundbox, t_loss, t_score_loss, t_reg_loss, t_cen_loss= model.predict(seq_torch, summary, reg_label, cen_label, maski)
                loss += t_loss
                score_loss += t_score_loss
                reg_loss += t_reg_loss
                cen_loss += t_cen_loss

                pred_boundbox = np.clip(pred_boundbox, 0, seq_len).round().astype(np.int32)

                score_pred2, pred_boundbox = nms(
                    score_pred.squeeze(0),
                    pred_boundbox,
                    nms_thresh
                )

                pred_summ, t_logits = boundbox_to_summary(
                    seq_len,
                    score_pred2,
                    pred_boundbox,
                    cps,
                    n_frames,
                    nfps,
                    picks
                )
                
                fscore, kTau, sRho = evaluate_summary(pred_summ[i], user_summary[i], score_pred[i], gtscore, eval_method='avg')
                if cps[0][-1][1] != n_frames[0]:
                    cps[0][-1][1] += 1
                    nfps[0][-1] += 1
                WSE, CIS, WIR, IR = evaluate_knapsack_opt(score_pred[i], gtscore[i], user_summary[i], cps[i], n_frames, nfps[i], picks[i])

                eval_metric = 'avg' if 'tvsum' in test_key else 'max'
                # tmp_fscore.append(get_summ_f1score(pred_summ, user_summary[i]))
                tmp_fscore.append(fscore)
                sRho_history.append(sRho)
                kTau_history.append(kTau)
                WIR_history.append(WIR)
                WSE_history.append(WSE)
                IR_history.append(IR)
                CIS_history.append(CIS)
                
                logits.append(t_logits)

                gtscore = gtscore.squeeze(0)

                gt_seg_score = generate_mrsum_seg_scores(torch.Tensor(gtscore), uniform_clip=5)
                gt_top50_summary = top50_summary(torch.Tensor(gt_seg_score))
                gt_top15_summary = top15_summary(torch.Tensor(gt_seg_score))

                score_pred = score_pred.squeeze(0)

                highlight_seg_machine_score = torch.Tensor(generate_mrsum_seg_scores(torch.Tensor(score_pred), uniform_clip=5))
                highlight_seg_machine_score = torch.exp(highlight_seg_machine_score) / (torch.exp(highlight_seg_machine_score).sum() + 1e-7)
                
                clone_machine_summary = highlight_seg_machine_score.clone().detach().cpu()
                clone_machine_summary = clone_machine_summary.numpy()

                aP50 = average_precision_score(gt_top50_summary, clone_machine_summary)
                aP15 = average_precision_score(gt_top15_summary, clone_machine_summary)

                map50_history.append(aP50)
                map15_history.append(aP15)


            if eval_metric == 'avg':
                final_f1 = np.mean(tmp_fscore)
            elif eval_metric == 'max':
                final_f1 = np.max(tmp_fscore)
            else:
                final_f1 = 0.
                print('Eval metric error.')
            logits_dict[test_key[0]] = logits

            stats.update(fscore=final_f1)

    final_map50 = np.mean(map50_history)
    final_map15 = np.mean(map15_history)
    final_sRho = np.mean(sRho_history)
    final_kTau = np.mean(kTau_history)
    final_WIR = np.mean(WIR_history)
    final_WSE = np.mean(WSE_history)
    final_IR = np.mean(IR_history)
    final_CIS = np.mean(CIS_history)
    return stats.fscore, loss, score_loss, reg_loss, cen_loss, logits_dict, final_map50, final_map15, final_sRho, final_kTau, final_WIR, final_WSE, final_IR, final_CIS


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
        picks
):
    score = np.zeros(seq_len, dtype=np.float32)

    for idx in range(len(pred_boundbox)):
        lo, hi = pred_boundbox[idx, 0], pred_boundbox[idx, 1]
        score[lo:hi] = np.maximum(score[lo:hi], [score_pred[idx]])

    pred_summ = get_keyshot_summ(np.expand_dims(score, axis=0), change_points, n_frames, nfps, picks)
    return pred_summ, score


def get_keyshot_summ(
        pred,
        cps,
        n_frames,
        nfps,
        picks,
        proportion=0.15
):

    # assert np.array(pred).shape == np.array(picks).shape
    # picks = np.asarray(picks, dtype=np.int32)

    # frame_scores = np.zeros(n_frames, dtype=np.float32)
    # for i in range(len(picks)):
    #     pos_lo = picks[i]
    #     pos_hi = picks[i + 1] if i + 1 < len(picks) else n_frames
    #     frame_scores[pos_lo:pos_hi] = pred[i]

    # seg_scores = np.zeros(len(cps), dtype=np.int32)
    # for seg_idx, (first, last) in enumerate(cps):
    #     scores = frame_scores[first:last + 1]
    #     seg_scores[seg_idx] = int(1000 * scores.mean())

    # limits = int(n_frames * proportion)
    # packed = knapsack(seg_scores, nfps, limits)

    # summary = np.zeros(n_frames, dtype=np.bool_)
    # for seg_idx in packed:
    #     first, last = cps[seg_idx]
    #     summary[first:last + 1] = True
    B, pred_len = np.array(pred).shape
    picks_len = picks[0].size

    batch_summary = []
    for b in range(B):
        frame_scores = copy.deepcopy(pred[b])
        # Assign scores to video shots as the average of the frames.
        seg_scores = np.zeros(len(cps[b]), dtype=np.int32)
        for seg_idx, (first, last) in enumerate(cps[b]):
            scores = frame_scores[first:last + 1]
            seg_scores[seg_idx] = int(1000 * scores.mean())

        # Apply knapsack algorithm to find the best shots
        limits = int(n_frames[b] * proportion)
        # packed = knapsack(seg_scores, nfps[b], limits)
        packed = knapSack(limits, nfps[b], seg_scores, len(nfps[b]))

        # Get key-shot based summary
        summary = np.zeros(n_frames[b], dtype=bool)
        for seg_idx in packed:
            first, last = cps[b][seg_idx]
            summary[first:last + 1] = True
        batch_summary.append(summary)
        # print(summary.shape)

    # batch_summary = np.array(batch_summary)

    zero_pad_batch_summary = np.array(list(zip_longest(*batch_summary, fillvalue=0))).T

    return zero_pad_batch_summary


# def knapsack(
#         values,
#         weights,
#         capacity
# ):

#     knapsack_solver = KnapsackSolver(KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER, 'test')

#     values = list(values)
#     weights = list(weights)
#     capacity = int(capacity)

#     knapsack_solver.Init(values, [weights], [capacity])
#     knapsack_solver.Solve()
#     packed_items = [x for x in range(0, len(weights)) if knapsack_solver.BestSolutionContains(x)]

#     return packed_items


def get_summ_f1score(
        pred_summ,
        test_summ
):

    pred_summ = np.asarray(pred_summ, dtype=np.bool_)
    test_summ = np.asarray(test_summ, dtype=np.bool_)
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
    max_len = max(pred.shape[1], test.shape[1])
    pred_copy = np.zeros(max_len, dtype=int)
    test_copy = np.zeros(max_len, dtype=int)
    test_copy[:test.shape[1]] = test
    test_copy = np.expand_dims(test_copy, 0)

    assert pred.shape == test_copy.shape
    

    pred = np.asarray(pred, dtype=np.bool_)
    test = np.asarray(test_copy, dtype=np.bool_)
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
    offsets_list = []

    B = len(summary)
    for i in range(B):
        seq_len, = summary[i].shape

        boundbox = seq_to_boundbox(summary[i])
        offsets = boundbox_to_offset(boundbox, seq_len)
        offsets_list.append(offsets)
    return offsets_list


def seq_to_boundbox(sequence):
    sequence = np.asarray(sequence, dtype=np.bool_)
    selected_indices, = np.where(sequence == 1)

    boundbox_lr = []
    for k, g in groupby(enumerate(selected_indices), lambda x: x[0] - x[1]):
        segment = list(map(itemgetter(1), g))
        start_frame, end_frame = segment[0], segment[-1] + 1
        boundbox_lr.append([start_frame, end_frame])

    boundbox_lr = np.asarray(boundbox_lr, dtype=np.int32)
    return boundbox_lr


def boundbox_to_offset(boundbox, seq_len):

    pos_idx = np.arange(seq_len, dtype=np.float32)
    offsets = np.zeros((seq_len, 2), dtype=np.float32)

    for lo, hi in boundbox:
        boundbox_pos = pos_idx[lo:hi]
        offsets[lo:hi] = np.vstack((boundbox_pos - lo, hi - 1 - boundbox_pos)).T

    return offsets


def get_cen_label(
        summary,
        offset,
        eps=1e-8
):
    cen_list = []
    B = len(summary)
    for i in range(B):
        summary[i] = np.asarray(summary[i], dtype=np.bool_)
        cen_label = np.zeros(summary[i].shape, dtype=np.float32)

        offset_left, offset_right = offset[i][summary[i], 0], offset[i][summary[i], 1]

        cen_label[summary[i]] = np.minimum(offset_left, offset_right) / (np.maximum(offset_left, offset_right) + eps)
        cen_list.append(cen_label)

    return cen_list


def sum_score_loss(score_pred, target, mask):

    target = target.type(torch.long)
    num_pos = torch.sum(target, axis=1)

    score_pred = score_pred.unsqueeze(-1)
    score_pred = torch.cat([1 - score_pred, score_pred], dim=-1)

    # print(score_pred.shape)
    # print(target.shape)

    loss = focal_loss(score_pred, target, mask)

    loss = loss / num_pos
    return torch.mean(loss)


def focal_loss(
        x,
        y,
        mask,
        alpha=0.25,
        gamma=2
):

    B, d, num_classes = x.shape
    fl_results = []
    t = one_hot_embedding(y, num_classes)
    for b in range(B):
        maski = mask[b].float().expand(num_classes, d).transpose(1, 0)
        xi = x[b]
        ti = t[b]

        p_t = xi * ti + (1 - xi) * (1 - ti)
        alpha_t = alpha * ti + (1 - alpha) * (1 - ti)
        #p_t += 0.001
        fl = -alpha_t * (1 - p_t).pow(gamma) * p_t.log()
        fl = torch.mul(fl, maski)
        fl = fl.sum()
        fl_results.append(fl)

    return torch.stack(fl_results)


def one_hot_embedding(labels, num_classes):

    eye = torch.eye(num_classes, device=labels.device)
    return eye[labels]


def sum_reg_loss(
        reg_pred,
        reg_label,
        target,
        eps: float = 1e-8
):

    target = target.type(torch.bool)
    reg_pred = reg_pred[target]
    reg_label = reg_label[target]

    iou = iou_offset(reg_pred, reg_label)
    loss = -torch.log(iou + eps).mean()

    return loss


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
    target = target.type(torch.bool)

    cen_pred = cen_pred[target]
    cen_label = cen_label[target]

    loss = F.binary_cross_entropy(cen_pred, cen_label)
    return loss

def generate_mrsum_seg_scores(cp_frame_scores, uniform_clip=5):
    # split in uniform division
    splits = torch.split(cp_frame_scores, uniform_clip)
    averages = [torch.mean(sp) for sp in splits]

    # segment_scores = torch.cat(averages)
    
    return averages

def top50_summary(scores):
    sort_idx = torch.argsort(scores, descending=True)
    # take the 50% shots
    median_index = len(scores) // 2 
    filtered_sort_idx = sort_idx[:median_index]
    selected_segs = [0] * len(scores)
    for index in filtered_sort_idx:
        selected_segs[index] = 1
    
    return selected_segs

def top15_summary(scores):
    sort_idx = torch.argsort(scores, descending=True)

    # take the 15% shots
    filter_index = int(len(scores) * 0.15) 
    filtered_sort_idx = sort_idx[:filter_index]
    selected_segs = [0] * len(scores)
    for index in filtered_sort_idx:
        selected_segs[index] = 1
    
    return selected_segs