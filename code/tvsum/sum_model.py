import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from functools import partial
import collections.abc
from itertools import repeat
import warnings
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from loc_nets import AttentionExtractor
import loc_train


class Model(nn.Module):
    def __init__(
            self,
            num_feature,
            num_hidden,
            num_head,
            depth,
            mlp_ratio,
            hidden_dim,
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            act_layer,
            scan_range,
            anchor_scales,
            device,
            seq_blocks=False,
            linear_op=False,
    ):
        super().__init__()
        self.num_hidden = num_hidden
        # self.init_weights()
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        self.attention = AttentionExtractor(num_head, num_feature, hidden_dim)

        if seq_blocks:
            self.feat_drop = nn.Dropout(p=0.4)
            norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
            self.norm = norm_layer(num_feature)

            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            self.blocks = nn.Sequential(
                *[
                    Block(
                        dim=num_feature,
                        num_heads=num_head,
                        hidden_dim=hidden_dim,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        act_layer=act_layer
                )
                for i in range(depth)
                ]
            )

        self.seq_blocks = seq_blocks
        self.linear_op = linear_op
        self.roi_poolings = [nn.AvgPool1d(scale, stride=1, padding=scale // 2)
                             for scale in anchor_scales]

        if linear_op:

            self.pre_linear = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(num_feature, num_feature)),
                ('act', nn.ReLU())
            ]))

        self.layer_norm = nn.LayerNorm(num_feature)
        self.feat_linear = nn.Sequential(
            nn.Linear(num_feature, num_hidden),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden)
        )

        self.scan_range = scan_range
        self.device = device

        self.linear_score = nn.Linear(num_hidden, 1)
        self.linear_reg = nn.Linear(num_hidden, 2)
        self.linear_zeta = nn.Linear(num_hidden, 1)

    def init_weights(self):
        self.apply(_init_weights)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        if self.seq_blocks:
            x = self.feat_drop(x)
            x = self.blocks(x)

        out = self.attention(x)
        out = out + x

        if self.linear_op:
            out = self.norm(out)
            out = self.pre_linear(out)

        out = self.layer_norm(out)

        roi_ori = self.roi_process(out, seq_len)

        score_pred = self.linear_score(roi_ori).sigmoid()

        score_pred = score_pred.view(seq_len, self.num_scales)
        reg_pred = self.linear_reg(roi_ori).view(seq_len, self.num_scales, 2)

        zeta = torch.squeeze(self.linear_zeta(roi_ori), 2)
        gaussian_map = self.get_gaussian(self.scan_range)

        pred_prop = []
        for scale in range(0, len(self.anchor_scales)):
            prop = self.propagation(
                score_pred[:, scale],
                roi_ori[:, scale, :],
                zeta[:, scale],
                gaussian_map,
                self.scan_range,
                self.device
            )
            pred_prop.append(prop)

        return score_pred, reg_pred, pred_prop

    def evaluate(self, x):
        batch_size, seq_len, _ = x.shape

        if self.seq_blocks:
            x = self.feat_drop(x)
            x = self.blocks(x)

        out = self.attention(x)
        out = out + x

        if self.linear_op:
            out = self.norm(out)
            out = self.pre_linear(out)

        out = self.layer_norm(out)

        roi_ori = self.roi_process(out, seq_len)

        score_pred = self.linear_score(roi_ori).sigmoid()

        score_pred = score_pred.view(seq_len, self.num_scales)
        reg_pred = self.linear_reg(roi_ori).view(seq_len, self.num_scales, 2)

        return score_pred, reg_pred


    def roi_process(self, input, seq_len):

        out = input.transpose(2, 1)

        pool_results = [roi_pooling(out) for roi_pooling in self.roi_poolings]
        out = torch.cat(pool_results, dim=0).permute(2, 0, 1)[:-1]

        output = self.feat_linear(out)

        return output


    def propagation(
            self,
            score_pred,
            features,
            zeta,
            gaussian_map,
            scan_range,
            device,
            scan_alp=0.5
    ):
        features = torch.squeeze(features, 0)
        score_pred = torch.squeeze(score_pred, 0)
        seq_len, _ = features.shape
        pred_prop = torch.zeros([seq_len], device=device)
        for index in range(seq_len):
            target = features[index]

            f_set = features[max(index - scan_range, 0): min(index + scan_range + 1, seq_len)]
            target = torch.unsqueeze(target, 0)
            target = target.repeat(f_set.shape[0], 1)
            sim = torch.cosine_similarity(target, f_set, 1)

            set_score = score_pred[max(index - scan_range, 0): min(index + scan_range + 1, seq_len)]
            sum_score = (sim * set_score).sum()
            sum_score = sum_score * scan_alp + score_pred[index]
            pred_prop[index] = sum_score

        pred_prop_ag = pred_prop.clone()
        for index in range(seq_len):
            s_range = pred_prop[max(index - scan_range, 0): min(index + scan_range + 1, seq_len)]
            arg = torch.argmax(s_range)
            offset = arg - min(scan_range, index)
            map_value = pred_prop[index] * gaussian_map[offset.item()]
            pred_prop_ag[index] = map_value
        pred_prop_ag = pred_prop_ag + zeta

        return pred_prop_ag

    def get_gaussian(self, scan_range, g_miu=0, g_sigma=4):
        gaussian_map = {}
        for off in range(-scan_range, scan_range + 1):
            gaussian_map[off] = np.exp(-np.power(off - g_miu, 2) / (2 * np.power(g_sigma, 2)))
        return gaussian_map

    def predict(self, seq, label_sum, reg_label, eps=1e-8):
        seq_len = seq.shape[1]
        score_pred, reg_pred = self.evaluate(seq)

        """"""
        # for calc val loss
        score_loss = loc_train.sum_score_loss(score_pred, label_sum)
        reg_loss = loc_train.sum_reg_loss(reg_pred, reg_label, label_sum)
        val_loss = score_loss + reg_loss

        """"""
        score_pred = score_pred.cpu().numpy().reshape(-1)
        reg_pred = reg_pred.cpu().numpy().reshape((-1, 2))


        anchors = loc_train.get_anchors(seq_len, self.anchor_scales)
        anchors = anchors.reshape((-1, 2))

        pred_boudbox = offset_to_boundbox(reg_pred, anchors)
        pred_boudbox = loc_train.cw_to_lr(pred_boudbox)

        return score_pred, pred_boudbox, val_loss.item(), score_loss.item(), reg_loss.item()



class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            hidden_dim=1024,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=None,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else DropPath(0.)
        act_layer = act_layer or nn.ReLU
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            hidden_dim=1024,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.hidden_dim = hidden_dim
        self.qkv = nn.Linear(dim, self.hidden_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.hidden_dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.hidden_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.hidden_dim)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()

        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(
            x,
            self.drop_prob,
            self.training
        )


def drop_path(
        x,
        drop_prob=0.,
        training=False
):

    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()

    output = x.div(keep_prob) * random_tensor

    return output


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.ReLU,
            drop=0.
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)

        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)

        x = self.act(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.drop2(x)

        return x


def xavier_init(module):
    cls_name = module.__class__.__name__
    if 'Linear' in cls_name or 'Conv' in cls_name:
        nn.init.xavier_uniform_(module.weight, gain=np.sqrt(2.0))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.1)

def _init_weights(
        module,
        name='',
        head_bias=0.,
        jax_impl=False
):

    if isinstance(module, nn.Linear):

        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)

        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)

        else:

            if jax_impl:
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)

            else:
                trunc_normal_(module.weight, std=.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    elif jax_impl and isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def lecun_normal_(tensor):
    variance_scaling_(
        tensor,
        mode='fan_in',
        distribution='truncated_normal'
    )


def variance_scaling_(
        tensor,
        scale=1.0,
        mode='fan_in',
        distribution='normal'
):

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def trunc_normal_(
        tensor,
        mean=0.,
        std=1.,
        a=-2.,
        b=2.
):
    return _no_grad_trunc_normal_(
        tensor,
        mean,
        std,
        a,
        b
    )


def _no_grad_trunc_normal_(
        tensor,
        mean,
        std,
        a,
        b
):

    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def perm_func(out, perm, device, coff=1):
    if not perm:
        return out
    else:
        p = torch.rand(out.shape, dtype=torch.float32, device=device)
        return out + coff * p


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x

        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def offset_to_boundbox(offsets, anchors):

    offsets = offsets.reshape(-1, 2)
    anchors = anchors.reshape(-1, 2)

    offset_center, offset_width = offsets[:, 0], offsets[:, 1]
    anchor_center, anchor_width = anchors[:, 0], anchors[:, 1]

    boundbox_center = offset_center * anchor_width + anchor_center
    boundbox_width = np.exp(offset_width) * anchor_width

    boundbox = np.vstack((boundbox_center, boundbox_width)).T

    return boundbox


