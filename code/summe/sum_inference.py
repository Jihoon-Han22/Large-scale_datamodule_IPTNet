
import os
import torch
import numpy as np
from pathlib import Path
import h5py
import random
import json

from sum_model import Model
import loc_train
from helper import Average, TensorboardWriter
import pdb
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader



def train(args, index, split, save_path, log_dir):

    # val_loss_bucket = []
    # val_f1_bucket = []
    # val_map50_bucket = []
    # val_map15_bucket = []

    model = Model(
        num_feature=args.num_feature,
        num_hidden=args.num_hidden, # check
        num_head=args.num_head, 
        depth=args.depth, # check
        mlp_ratio=args.mlp_ratio, #check
        hidden_dim=args.hidden_dim,  #여기가 parameter 줄음
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=None,
        act_layer=None,
        scan_range=args.scan_range,  #여기
        device=args.device,
        seq_blocks=args.seq_blocks  # check
    )

    model_tech = Model(
        num_feature=args.num_feature,
        num_hidden=args.num_hidden,
        num_head=args.num_head,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio,
        hidden_dim=args.hidden_dim,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=None,
        act_layer=None,
        scan_range=args.scan_range,
        device=args.device,
        seq_blocks=args.seq_blocks
    )

    print('----------------------------')
    all_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params_Mb = all_params / 1024**2
    trainable_params_Mb = trainable_params / 1024**2
    print(f'# of Trainable parameters : {trainable_params}\t {round(trainable_params_Mb,2)}MB')
    print(f'# of All parameters (including non-trainable) : {all_params}\t {round(all_params_Mb,2)}MB')
    print('----------------------------')

    model = model.to(args.device)
    model_tech = model_tech.to(args.device)

    writer = TensorboardWriter(log_dir)
    global_step = 0

    max_val_fscore = -1

    #train_set = Dataset(split['train_keys'])
    train_set = Dataset('train')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=BatchCollator())

    #val_set = Dataset(split['val_keys'])
    val_set = Dataset('val')
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=BatchCollator())

    #test_set = Dataset(split['test_keys'])
    test_set = Dataset('test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=BatchCollator())

    loc_model, loc_model_tech = loc_train.init_loc(args)

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer1 = torch.optim.Adam(params)

    params = []
    for key, value in loc_model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.loc_lr, "weight_decay": args.weight_decay}]
    optimizer2 = torch.optim.Adam(params)

    # # Resume training
    # model.load_state_dict(torch.load(os.path.join(save_path, 'model_sum.pt')))
    # model_tech.load_state_dict(torch.load(os.path.join(save_path, 'model_sum_tech.pt')))
    # loc_model.load_state_dict(torch.load(os.path.join(save_path, 'model_loc.pt')))
    # loc_model_tech.load_state_dict(torch.load(os.path.join(save_path, 'model_loc_tech.pt')))


    # model, model_tech, optimizer1, optimizer2, loc_model, loc_model_tech, max_val_fscore, writer, global_step = loc_train.train_loc(
    #     args,
    #     model,
    #     model_tech,
    #     optimizer1,
    #     optimizer2,
    #     loc_model,
    #     loc_model_tech,
    #     False,
    #     args.set_epoch * 2,
    #     train_loader,
    #     val_loader,
    #     max_val_fscore,
    #     save_path,
    #     writer,
    #     global_step
    # )

    # for epoch in range(30, args.max_epoch):
    #     model.train()
    #     model_tech.train()
    #     stats = Average('loss', 'score_loss', 'reg_loss', 'cen_loss')
        

    #     num_batches = int(len(train_loader))
    #     iterator = iter(train_loader)
    #     loss_bucket = []

    #     for _ in tqdm(range(num_batches)):
            
    #         data = next(iterator)
    #         # _, video_seq, gtscore, change_points, n_frames, nfps, picks, _ = next(iterator)

    #         video_seq = data['features']
    #         gtscore = data['gtscore']
    #         change_points = data['change_points']
    #         n_frames = data['n_frames']
    #         nfps = data['n_frame_per_seg']
    #         picks = data['picks']
    #         mask = data['mask']

    #         mask = mask.to(args.device)

    #         summary = loc_train.get_keyshot_summ(gtscore, change_points, n_frames, nfps, picks)
            
    #         # summary = keyshot_summ[picks]
    #         # summary = loc_train.downsample_summ(keyshot_summ)
    #         # summary = [keyshot_summ[i] for i in picks]

    #         if not summary.any():
    #             continue

    #         video_seq = torch.tensor(video_seq, dtype=torch.float32).to(args.device)

    #         reg_label = loc_train.get_reg_label(summary)
    #         cen_label = loc_train.get_cen_label(summary, reg_label)


    #         score_pred, reg_pred, cen_pred, prop_pred = model(video_seq)

    #         summary = summary.astype(np.float32)

    #         summary = torch.tensor(summary, dtype=torch.float32).to(args.device)
    #         reg_label = torch.tensor(reg_label, dtype=torch.float32).to(args.device)
    #         cen_label = torch.tensor(cen_label, dtype=torch.float32).to(args.device)

    #         score_pred = torch.squeeze(score_pred, 0)
    #         reg_pred = torch.squeeze(reg_pred, 0)
    #         cen_pred = torch.squeeze(cen_pred, 0)

    #         #print(score_pred.shape)
    #         #print(mask.shape)

    #         score_loss = loc_train.sum_score_loss(score_pred, summary, mask)
    #         reg_loss = loc_train.sum_reg_loss(reg_pred, reg_label, summary)
    #         cen_loss = loc_train.sum_cen_loss(cen_pred, cen_label, summary)

    #         loss = score_loss + reg_loss + cen_loss
    #         writer.update_loss(loss.item(), global_step, 'train/loss')
    #         writer.update_loss(score_loss.item(), global_step, 'train/score_loss')
    #         writer.update_loss(reg_loss.item(), global_step, 'train/reg_loss')
    #         writer.update_loss(cen_loss.item(), global_step, 'train/cen_loss')

    #         optimizer1.zero_grad()
    #         loss.backward()
    #         optimizer1.step()
    #         global_step += 1

    #         for tech_param, param in zip(model_tech.parameters(), model.parameters()):
    #             tech_param.data.mul_(args.ema).add_(1 - args.ema, param.data)

    #         stats.update(loss=loss.item(), score_loss=score_loss.item(), reg_loss=reg_loss.item(), cen_loss=cen_loss.item())
    #         loss_bucket.append(loss.item())

    #     val_fscore, val_loss, val_score_loss, val_reg_loss, val_cen_loss, val_logits, map50, map15 = loc_train.sum_eval(model, val_loader, args.nms_thresh, args.device)
    #     val_loss_bucket.append(val_loss)
    #     val_f1_bucket.append(val_fscore)
    #     val_map50_bucket.append(map50)
    #     val_map15_bucket.append(map15)

    #     writer.update_loss(val_fscore, global_step, 'val/f1_epoch')
    #     writer.update_loss(val_loss, global_step, 'val/loss')
    #     writer.update_loss(val_score_loss, global_step, 'val/score_loss')
    #     writer.update_loss(val_reg_loss, global_step, 'val/reg_loss')
    #     writer.update_loss(val_cen_loss, global_step, 'val/cen_loss')

    #     val_fscore_tech, loss, score_loss, reg_loss, cen_loss, val_tech_logits, map50, map15 = loc_train.sum_eval(model_tech, val_loader, args.nms_thresh, args.device)

    #     if max_val_fscore < val_fscore:
    #         max_val_fscore = val_fscore
    #         torch.save(model.state_dict(), os.path.join(save_path, 'model_sum_best.pt'))
    #     if max_val_fscore < val_fscore_tech:
    #         max_val_fscore = val_fscore_tech
    #         torch.save(model_tech.state_dict(), os.path.join(save_path, 'model_sum_best.pt'))

    #     torch.save(model.state_dict(), os.path.join(save_path, 'model_sum.pt'))
    #     torch.save(model_tech.state_dict(), os.path.join(save_path, 'model_sum_tech.pt'))
        
    #     val_loss_path = os.path.join(log_dir, 'val_loss.pt')
    #     val_f1_path = os.path.join(log_dir, 'val_f1.pt')
    #     torch.save(val_loss_bucket, val_loss_path)
    #     torch.save(val_f1_bucket, val_f1_path)
    #     writer.update_loss(max_val_fscore, global_step, 'val/f1_best')
    #     writer.update_loss(stats.loss, global_step, 'train/loss_epoch')
        
    #     print(f'Epoch: {epoch} Train Loss: {np.mean(np.array(loss_bucket)):.4f}')
    #     print(f'Val Loss: {stats.loss:.4f}')
    #     print(f'val fscore: {val_fscore:.4f}')
    #     print(f'val Max fscore: {max_val_fscore:.4f}')
    #     print(f'val mAP50: {map50:.4f}, val mAP15: {map15:.4f}')

        # best_model = Model(
        # num_feature=args.num_feature,
        # num_hidden=args.num_hidden,
        # num_head=args.num_head,
        # depth=args.depth,
        # hidden_dim=args.hidden_dim,
        # mlp_ratio=args.mlp_ratio,
        # qkv_bias=True,
        # drop_rate=0.,
        # attn_drop_rate=0.,
        # drop_path_rate=0.,
        # norm_layer=None,
        # act_layer=None,
        # scan_range=args.scan_range,
        # device=args.device,
        # seq_blocks=args.seq_blocks
        # )
        # best_model = best_model.to(args.device)
        # best_model.load_state_dict(torch.load(os.path.join(save_path, 'model_sum_best.pt')))
        # test_fscore, test_loss, _, _, _, test_logits, final_map50, final_map15 = loc_train.sum_eval(best_model, test_loader, args.nms_thresh, args.device)
        # test_logits_path = os.path.join(log_dir, 'test_logits.pt')
        # torch.save(test_logits, test_logits_path)

        # print()
        # print(f'Epoch: {epoch}')
        # print(f'Test score is : {test_fscore:.4f}%')
        # print(f'Final map50 is : {final_map50:.4f}%')
        # print(f'Final map15 is : {final_map15:.4f}%')

        # if epoch % args.coll_step == 0:

        #     model, model_tech, optimizer1, optimizer2, loc_model, loc_model_tech, max_val_fscore, writer, global_step = loc_train.train_loc(
        #         args,
        #         model,
        #         model_tech,
        #         optimizer1,
        #         optimizer2,
        #         loc_model,
        #         loc_model_tech,
        #         True,
        #         args.set_epoch,
        #         train_loader,
        #         val_loader,
        #         max_val_fscore,
        #         save_path,
        #         writer,
        #         global_step
        #     )

        #     val_fscore, val_loss, val_score_loss, val_reg_loss, val_cen_loss, val_logits, map50, map15 = loc_train.sum_eval(model, val_loader, args.nms_thresh, args.device)
        #     val_loss_bucket.append(val_loss)
        #     val_f1_bucket.append(val_fscore)
        #     val_map50_bucket.append(map50)
        #     val_map15_bucket.append(map15)

        #     writer.update_loss(val_fscore, global_step, 'val/f1_epoch')
        #     writer.update_loss(val_loss, global_step, 'val/loss')
        #     writer.update_loss(val_score_loss, global_step, 'val/score_loss')
        #     writer.update_loss(val_reg_loss, global_step, 'val/reg_loss')
        #     writer.update_loss(val_cen_loss, global_step, 'val/cen_loss')

        #     val_fscore_tech, loss, score_loss, reg_loss, cen_loss, val_tech_logits, map50, map15 = loc_train.sum_eval(model_tech, val_loader, args.nms_thresh, args.device)

        #     if max_val_fscore < val_fscore:
        #         max_val_fscore = val_fscore
        #         torch.save(model.state_dict(), os.path.join(save_path, 'model_sum_best.pt'))
        #     if max_val_fscore < val_fscore_tech:
        #         max_val_fscore = val_fscore_tech
        #         torch.save(model_tech.state_dict(), os.path.join(save_path, 'model_sum_best.pt'))

        #     torch.save(model.state_dict(), os.path.join(save_path, 'model_sum.pt'))
        #     torch.save(model_tech.state_dict(), os.path.join(save_path, 'model_sum_tech.pt'))
        
        #     val_loss_path = os.path.join(log_dir, 'val_loss.pt')
        #     val_f1_path = os.path.join(log_dir, 'val_f1.pt')
        #     torch.save(val_loss_bucket, val_loss_path)
        #     torch.save(val_f1_bucket, val_f1_path)
        #     writer.update_loss(max_val_fscore, global_step, 'val/f1_best')
        #     writer.update_loss(stats.loss, global_step, 'train/loss_epoch')

        #     print(f'Epoch: {epoch}')
        #     print(f'Val Loss: {round(stats.loss, 4)}')
        #     print(f'val fscore: {round(val_fscore, 4)}')
        #     print(f'val Max fscore: {round(max_val_fscore, 4)}')
        #     print(f'val mAP50: {round(map50, 4)}, val mAP15: {round(map15, 4)}')

    # if (epoch+1) % 5 ==0:
    #     best_model = Model(
    #         num_feature=args.num_feature,
    #         num_hidden=args.num_hidden,
    #         num_head=args.num_head,
    #         depth=args.depth,
    #         hidden_dim=args.hidden_dim,
    #         mlp_ratio=args.mlp_ratio,
    #         qkv_bias=True,
    #         drop_rate=0.,
    #         attn_drop_rate=0.,
    #         drop_path_rate=0.,
    #         norm_layer=None,
    #         act_layer=None,
    #         scan_range=args.scan_range,
    #         device=args.device,
    #         seq_blocks=args.seq_blocks
    #     )
    #     best_model = best_model.to(args.device)
    #     best_model.load_state_dict(torch.load(str(save_path)))
    #     test_fscore, test_loss, _, _, _, test_logits, final_map50, final_map15 = loc_train.sum_eval(best_model, test_loader, args.nms_thresh, args.device)
    #     test_logits_path = os.path.join(log_dir, 'test_logits.pt')
    #     torch.save(test_logits, test_logits_path)

        # print(f'Test f1score is : {test_fscore:.4f}%')
        # print(f'Test map50 is : {final_map50:.4f}%')
        # print(f'Test map15 is : {final_map15:.4f}%')
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
        device=args.device,
        seq_blocks=args.seq_blocks
    )
    best_model = best_model.to(args.device)
    best_model.load_state_dict(torch.load(os.path.join(save_path, 'model_sum_best.pt')))
    test_fscore, test_loss, _, _, _, test_logits, final_map50, final_map15, final_sRho, final_kTau, final_WIR, final_WSE, final_IR, final_CIS = loc_train.sum_eval(best_model, test_loader, args.nms_thresh, args.device)
    test_logits_path = os.path.join(log_dir, 'test_logits.pt')
    torch.save(test_logits, test_logits_path)
    print()
    print('----------------------------------------')
    print(f'Test F1 @ best val/f1 : {test_fscore}')
    print('----------------------------------------')
    # return max_val_fscore, test_fscore, final_map50, final_map15
    return test_fscore, final_map50, final_map15, final_sRho, final_kTau, final_WIR, final_WSE, final_IR, final_CIS


class Dataset():
    def __init__(self, mode):
        #self.keys = keys
        #self.datasets = self.get_datasets(keys)
        self.mode = mode
        self.dataset = '../../summarization_dataset/mrsum_with_features_gtsummary_modified.h5'
        self.split_file = '../../Mr.Sum/Mr.Sum/dataset/mrsum_split.json'

        self.video_data = h5py.File(self.dataset, 'r')

        with open(self.split_file, 'r') as f:
            self.data = json.loads(f.read())

    def __getitem__(self, index):
        video_name = self.data[self.mode + '_keys'][index]

        #key = self.keys[index]
        #video_path = Path(key)
        #dataset_name = str(video_path.parent)
        #video_name = video_path.name
        #video_file = self.datasets[dataset_name][video_name]
        d = {}

        # d['video_name'] = video_name
        # #seq = video_file['features'][...].astype(np.float32)
        # d['seq'] = np.array(self.video_data[video_name + '/features'])
        # #gtscore = video_file['gtscore'][...].astype(np.float32)
        # d['gtscore'] = np.array(self.video_data[video_name + '/gtscore'])
        # #cps = video_file['change_points'][...].astype(np.int32)
        # d['cps'] = np.array(self.video_data[video_name + '/change_points'])
        # #n_frames = video_file['n_frames'][...].astype(np.int32)
        # d['n_frames'] = np.array(self.video_data[video_name + '/features']).shape[0]
        # #nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
        # d['nfps'] = np.array([cp[1]-cp[0] for cp in d['cps']])
        # #picks = video_file['picks'][...].astype(np.int32)
        # d['picks'] = np.array([i for i in range(d['n_frames'])])
        d = {}
        d['video_name'] = video_name
        d['features'] = torch.Tensor(np.array(self.video_data[video_name + '/features']))
        d['gtscore'] = torch.Tensor(np.array(self.video_data[video_name + '/gtscore']))

        n_frames = d['features'].shape[0]
        cps = np.array(self.video_data[video_name + '/change_points'])
        d['n_frames'] = np.array(n_frames)
        d['picks'] = np.array([i for i in range(n_frames)])
        d['change_points'] = cps
        d['n_frame_per_seg'] = np.array([cp[1]-cp[0] for cp in cps])
        d['gt_summary'] = np.expand_dims(np.array(self.video_data[video_name + '/gt_summary']), axis=0)

        # user_summary = None
        # if 'user_summary' in video_file:
        #     user_summary = video_file['user_summary'][...].astype(np.float32)
        # d['user_summary'] = np.expand_dims(np.array(self.video_data[video_name + '/gt_summary']), axis=0)

        # d['gtscore'] -= d['gtscore'].min()
        # d['gtscore'] /= d['gtscore'].max()

        # d = {}
        # d['key'] = key
        # d['seq'] = seq
        # d['gtscore'] = gtscore
        # d['cps'] = cps
        # d['n_frames'] = n_frames
        # d['nfps'] = nfps
        # d['picks'] = picks
        # d['user_summary'] = user_summary

        return d

    def __len__(self):
        self.len = len(self.data[self.mode+'_keys'])
        return self.len

    @staticmethod
    def get_datasets(keys):
        dataset_paths = {str(Path(key).parent) for key in keys}
        datasets = {path: h5py.File(path, 'r') for path in dataset_paths}
        return datasets


class BatchCollator(object):
    def __call__(self, batch):
        video_name, features, gtscore= [],[],[]
        cps, nseg, n_frames, picks, gt_summary = [], [], [], [], []

        try:
            for data in batch:
                video_name.append(data['video_name'])
                features.append(data['features'])
                gtscore.append(data['gtscore'])
                cps.append(data['change_points'])
                nseg.append(data['n_frame_per_seg'])
                n_frames.append(data['n_frames'])
                picks.append(data['picks'])
                gt_summary.append(data['gt_summary'])
        except:
            print('Error in batch collator')

        lengths = torch.LongTensor(list(map(lambda x: x.shape[0], features)))
        max_len = max(list(map(lambda x: x.shape[0], features)))

        mask = torch.Tensor.bool(torch.arange(max_len)[None, :] < lengths[:, None])
        
        frame_feat = pad_sequence(features, batch_first=True)
        gtscore = pad_sequence(gtscore, batch_first=True)

        # batch_data = {'video_name' : video_name, 'features' : frame_feat, 'gtscore':gtscore, 'mask':mask}
        batch_data = {'video_name' : video_name, 'features' : frame_feat, 'gtscore':gtscore, 'mask':mask, \
                      'n_frames': n_frames, 'picks': picks, 'n_frame_per_seg': nseg, 'change_points': cps, \
                        'gt_summary': gt_summary, 'mask': mask}
        return batch_data
