
import os
import torch
import numpy as np
from pathlib import Path
import h5py
import random

from sum_model import Model
import loc_train
from helper import Average, TensorboardWriter


def inference(args, index, split, ckpt_path, log_dir):

    test_set = Dataset(split['test_keys'])
    test_loader = Loader(test_set, shuffle=False)

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
    best_model.load_state_dict(torch.load(str(ckpt_path)))
    test_fscore, test_loss, _, _, test_logits = loc_train.sum_eval(best_model, test_loader, args.nms_thresh, args.device, args=args)
    test_logits_path = os.path.join(log_dir, 'test_logits.pt')
    torch.save(test_logits, test_logits_path)


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


