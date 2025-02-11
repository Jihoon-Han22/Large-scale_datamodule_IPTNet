
import os
import torch
import numpy as np
from pathlib import Path
import h5py
import random

from sum_model import Model
import loc_train as loc_train
from helper import Average, TensorboardWriter
import sum_model
import warnings

warnings.filterwarnings("ignore")

def train(args, index, split, save_path, log_dir):
# def train(args, split, save_path, log_dir):
    val_loss_bucket = []
    val_f1_bucket = []
    model = Model(
        num_feature=args.num_feature,
        num_hidden=args.num_hidden,
        num_head=args.num_head,
        depth=args.depth,
        hidden_dim=args.hidden_dim,
        mlp_ratio=args.mlp_ratio,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=None,
        act_layer=None,
        scan_range=args.scan_range,
        anchor_scales=args.scales,
        device=args.device,
        seq_blocks=args.seq_blocks,
        linear_op=args.linear_op
    )

    model_tech = Model(
        num_feature=args.num_feature,
        num_hidden=args.num_hidden,
        num_head=args.num_head,
        depth=args.depth,
        hidden_dim=args.hidden_dim,
        mlp_ratio=args.mlp_ratio,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=None,
        act_layer=None,
        scan_range=args.scan_range,
        anchor_scales=args.scales,
        device=args.device,
        seq_blocks=args.seq_blocks,
        linear_op=args.linear_op
    )

    print('----------------------------')
    all_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params_Mb = all_params / 1024**2
    trainable_params_Mb = trainable_params / 1024**2
    print(f'# of Trainable parameters : {trainable_params}\t {round(trainable_params_Mb,2)}M')
    print(f'# of All parameters (including non-trainable) : {all_params}\t {round(all_params_Mb,2)}M')
    print('----------------------------')

    model = model.to(args.device)
    model_tech = model_tech.to(args.device)

    model.apply(sum_model.xavier_init)
    model_tech.apply(sum_model.xavier_init)

    writer = TensorboardWriter(log_dir)
    global_step = 0
    max_val_fscore = -1

    train_set = Dataset(split['train_keys'])
    train_loader = Loader(train_set, shuffle=True)

    val_set = Dataset(split['val_keys'])
    val_loader = Loader(val_set, shuffle=False)

    test_set = Dataset(split['test_keys'])
    test_loader = Loader(test_set, shuffle=False)

    loc_model, loc_model_tech = loc_train.init_loc(args)

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]

    for key, value in loc_model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.loc_lr}]
    optimizer = torch.optim.Adam(params)


    model, model_tech, optimizer, loc_model, loc_model_tech, max_val_fscore = loc_train.train_loc(
        args,
        model,
        model_tech,
        optimizer,
        loc_model,
        loc_model_tech,
        False,
        args.set_epoch * 2,
        train_loader,
        val_loader,
        max_val_fscore,
        save_path
    )

    for epoch in range(args.max_epoch):

        model.train()
        stats = Average('loss', 'score_loss', 'reg_loss')

        for _, seq, gtscore, change_points, n_frames, nfps, picks, _ in train_loader:
            keyshot_summ = loc_train.get_keyshot_summ(gtscore, change_points, n_frames, nfps, picks)
            summary = keyshot_summ[picks]
            # summary = loc_train.downsample_summ(keyshot_summ)

            if not summary.any():
                continue

            target_boundbox = loc_train.seq_to_boundbox(summary)
            target_boundbox = loc_train.lr_to_cw(target_boundbox)

            anchors = loc_train.get_anchors(summary.size, args.scales)
            label_sum, reg_label = loc_train.pos_sum(anchors, target_boundbox, args.pos_iou_thresh)

            num_pos = label_sum.sum()

            label_sum_neg, _ = loc_train.pos_sum(anchors, target_boundbox, args.neg_iou_thresh)
            label_sum_neg = loc_train.neg_sum(label_sum_neg, int(args.neg_sample_ratio * num_pos))

            label_sum_incom, _ = loc_train.pos_sum(anchors, target_boundbox, args.incomplete_iou_thresh)
            label_sum_incom[label_sum_neg != 1] = 1
            label_sum_incom = loc_train.neg_sum(label_sum_incom, int(args.incomplete_sample_ratio * num_pos))

            label_sum[label_sum_neg == -1] = -1
            label_sum[label_sum_incom == -1] = -1

            label_sum = torch.tensor(label_sum, dtype=torch.float32).to(args.device)
            reg_label = torch.tensor(reg_label, dtype=torch.float32).to(args.device)

            video_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)

            score_pred, reg_pred, prop_pred = model(video_seq)

            score_loss = loc_train.sum_score_loss(score_pred, label_sum)
            reg_loss = loc_train.sum_reg_loss(reg_pred, reg_label, label_sum)

            loss = score_loss + reg_loss
            writer.update_loss(loss.item(), global_step, 'train/loss')
            writer.update_loss(score_loss.item(), global_step, 'train/score_loss')
            writer.update_loss(reg_loss.item(), global_step, 'train/reg_loss')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            for tech_param, param in zip(model_tech.parameters(), model.parameters()):
                tech_param.data.mul_(args.ema).add_(1 - args.ema, param.data)

            stats.update(loss=loss.item(), score_loss=score_loss.item(), reg_loss=reg_loss.item())

        if epoch % args.coll_step == 0:

            model, model_tech, optimizer, loc_model, loc_model_tech, max_val_fscore = loc_train.train_loc(
                args,
                model,
                model_tech,
                optimizer,
                loc_model,
                loc_model_tech,
                True,
                args.set_epoch,
                train_loader,
                val_loader,
                max_val_fscore,
                save_path
            )

        val_fscore, val_loss, val_score_loss, val_reg_loss, val_logits = loc_train.sum_eval(model, val_loader, args.nms_thresh, args.device, args=args)
        val_loss_bucket.append(val_loss)
        val_f1_bucket.append(val_fscore)

        writer.update_loss(val_fscore, global_step, 'val/f1_epoch')
        writer.update_loss(val_loss, global_step, 'val/loss')
        writer.update_loss(val_score_loss, global_step, 'val/score_loss')
        writer.update_loss(val_reg_loss, global_step, 'val/reg_loss')

        val_fscore_tech, _, _, _, _ = loc_train.sum_eval(model_tech, val_loader, args.nms_thresh, args.device, args=args)

        if max_val_fscore < val_fscore:
            max_val_fscore = val_fscore
            torch.save(model.state_dict(), str(save_path))
        if max_val_fscore < val_fscore_tech:
            max_val_fscore = val_fscore_tech
            torch.save(model_tech.state_dict(), str(save_path))

        val_loss_path = os.path.join(log_dir, 'val_loss.pt')
        val_f1_path = os.path.join(log_dir, 'val_f1.pt')
        torch.save(val_loss_bucket, val_loss_path)
        torch.save(val_f1_bucket, val_f1_path)

        writer.update_loss(max_val_fscore, global_step, 'val/f1_best')
        writer.update_loss(stats.loss, global_step, 'train/loss_epoch')
        print(f'Epoch: {epoch} Loss: {round(stats.loss, 4)} F-score: {round(val_fscore, 4)} Max F-score: {round(max_val_fscore, 4)}')


    ### Test by best val/f1
    # load model -- save_path
    best_model = Model(
        num_feature=args.num_feature,
        num_hidden=args.num_hidden,
        num_head=args.num_head,
        depth=args.depth,
        hidden_dim=args.hidden_dim,
        mlp_ratio=args.mlp_ratio,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=None,
        act_layer=None,
        scan_range=args.scan_range,
        anchor_scales=args.scales,
        device=args.device,
        seq_blocks=args.seq_blocks,
        linear_op=args.linear_op
    )
    best_model = best_model.to(args.device)
    best_model.load_state_dict(torch.load(str(save_path)))
    test_fscore, test_loss, _, _, test_logits = loc_train.sum_eval(best_model, test_loader, args.nms_thresh, args.device, args=args)
    test_logits_path = os.path.join(log_dir, 'test_logits.pt')
    torch.save(test_logits, test_logits_path)
    return max_val_fscore, test_fscore



class Dataset():
    def __init__(self, keys):
        self.keys = keys
        self.datasets = self.get_datasets(keys)

    def __getitem__(self, index):
        key = self.keys[index]
        video_path = Path(key)
        dataset_name = str(video_path.parent)
        video_name = video_path.name
        video_file = self.datasets[dataset_name][video_name]

        seq = video_file['features'][...].astype(np.float32)
        gtscore = video_file['gtscore'][...].astype(np.float32)
        cps = video_file['change_points'][...].astype(np.int32)
        n_frames = video_file['n_frames'][...].astype(np.int32)
        nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
        picks = video_file['picks'][...].astype(np.int32)
        user_summary = None
        if 'user_summary' in video_file:
            user_summary = video_file['user_summary'][...].astype(np.float32)
        gtscore -= gtscore.min()
        gtscore /= gtscore.max()

        return key, seq, gtscore, cps, n_frames, nfps, picks, user_summary

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def get_datasets(keys):
        dataset_paths = {str(Path(key).parent) for key in keys}
        datasets = {path: h5py.File(path, 'r') for path in dataset_paths}
        return datasets


class Loader():
    def __init__(self, dataset, shuffle):
        self.dataset = dataset
        self.shuffle = shuffle
        self.data_idx = list(range(len(self.dataset)))

    def __iter__(self):
        self.iter_idx = 0
        if self.shuffle:
            random.shuffle(self.data_idx)
        return self

    def __next__(self):
        if self.iter_idx == len(self.dataset):
            raise StopIteration
        curr_idx = self.data_idx[self.iter_idx]
        batch = self.dataset[curr_idx]
        self.iter_idx += 1
        return batch


