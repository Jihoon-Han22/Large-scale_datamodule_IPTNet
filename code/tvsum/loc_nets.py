import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocNets(nn.Module):
    def __init__(self, args, word_vectors):
        super(LocNets, self).__init__()
        self.embedding_net = Embedding(
            num_words=args.num_words,
            num_chars=args.num_chars,
            out_dim=args.dim,
            word_dim=args.word_dim,
            char_dim=args.char_dim,
            word_vectors=word_vectors,
            drop_rate=args.drop_rate
        )
        self.video_affine = VisualProjection(
            visual_dim=args.visual_dim,
            dim=args.dim,
            drop_rate=args.drop_rate
        )
        self.feature_encoder = FeatureEncoder(
            dim=args.dim,
            num_heads=args.loc_heads,
            kernel_size=7,
            num_layers=4,
            max_pos_len=args.max_pos_len,
            drop_rate=args.drop_rate
        )
        self.cq_attention = CQAttention(
            dim=args.dim,
            drop_rate=args.drop_rate
        )
        self.cq_concat = CQConcatenate(
            dim=args.dim
        )
        self.highlight_layer = HighLightLayer(
            dim=args.dim
        )
        self.predictor = ConditionedPredictor(
            dim=args.dim
        )
        self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()
        self.apply(init_weights)

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):
        video_features = self.video_affine(video_features)
        query_features = self.embedding_net(word_ids, char_ids)
        video_features = self.feature_encoder(video_features, mask=v_mask)
        query_features = self.feature_encoder(query_features, mask=q_mask)
        features = self.cq_attention(video_features, query_features, v_mask, q_mask)
        features = self.cq_concat(features, query_features, q_mask)
        h_score = self.highlight_layer(features, v_mask)
        features = features * h_score.unsqueeze(2)
        start_logits, end_logits = self.predictor(features, mask=v_mask)
        return h_score, start_logits, end_logits

    def with_sum_data(self, video_features):
        video_features = self.video_affine(video_features)
        video_features = self.feature_encoder(video_features)
        h_score = self.highlight_layer(video_features, torch.ones(video_features.shape[:2]).to('cuda'))
        return h_score

    def compute_highlight_loss(self, scores, labels, mask):
        return self.highlight_layer.compute_loss(
            scores=scores,
            labels=labels,
            mask=mask
        )

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor.compute_cross_entropy_loss(
            start_logits=start_logits,
            end_logits=end_logits,
            start_labels=start_labels,
            end_labels=end_labels
        )


class Embedding(nn.Module):
    def __init__(self, num_words, num_chars, word_dim, char_dim, drop_rate, out_dim, word_vectors=None):
        super(Embedding, self).__init__()

        self.word_emb = WordEmbedding(
            num_words,
            word_dim,
            drop_rate,
            word_vectors=word_vectors
        )

        self.char_emb = CharacterEmbedding(
            num_chars,
            char_dim,
            drop_rate)

        self.linear = Conv1D(
            in_dim=word_dim + 100,
            out_dim=out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, word_ids, char_ids):
        word_emb = self.word_emb(word_ids)
        char_emb = self.char_emb(char_ids)
        emb = torch.cat([word_emb, char_emb], dim=2)
        emb = self.linear(emb)
        return emb


class WordEmbedding(nn.Module):
    def __init__(self, num_words, word_dim, drop_rate, word_vectors=None):
        super(WordEmbedding, self).__init__()

        self.is_pretrained = False if word_vectors is None else True
        if self.is_pretrained:
            self.pad_vec = nn.Parameter(torch.zeros(size=(1, word_dim), dtype=torch.float32), requires_grad=False)
            unk_vec = torch.empty(size=(1, word_dim), requires_grad=True, dtype=torch.float32)
            nn.init.xavier_uniform_(unk_vec)
            self.unk_vec = nn.Parameter(unk_vec, requires_grad=True)
            self.glove_vec = nn.Parameter(torch.tensor(word_vectors, dtype=torch.float32), requires_grad=False)
        else:
            self.word_emb = nn.Embedding(num_words, word_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, word_ids):
        if self.is_pretrained:
            word_emb = F.embedding(word_ids, torch.cat([self.pad_vec, self.unk_vec, self.glove_vec], dim=0),
                                   padding_idx=0)
        else:
            word_emb = self.word_emb(word_ids)
        return self.dropout(word_emb)


class CharacterEmbedding(nn.Module):
    def __init__(self, num_chars, char_dim, drop_rate):
        super(CharacterEmbedding, self).__init__()

        self.char_emb = nn.Embedding(num_chars, char_dim, padding_idx=0)
        kernels, channels = [1, 3, 3], [10, 30, 60]
        self.char_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=char_dim, out_channels=channel, kernel_size=(1, kernel), stride=(1, 1), padding=0,
                          bias=True),
                nn.ReLU()
            ) for kernel, channel in zip(kernels, channels)
        ])
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, char_ids):
        char_emb = self.char_emb(char_ids)
        char_emb = self.dropout(char_emb)
        char_emb = char_emb.permute(0, 3, 1, 2)
        char_outputs = []
        for conv_layer in self.char_convs:
            output = conv_layer(char_emb)
            output, _ = torch.max(output, dim=3, keepdim=False)
            char_outputs.append(output)
        char_output = torch.cat(char_outputs, dim=1)
        return char_output.permute(0, 2, 1)


class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        return x.transpose(1, 2)


class VisualProjection(nn.Module):
    def __init__(self, visual_dim, dim, drop_rate=0.0):
        super(VisualProjection, self).__init__()

        self.drop = nn.Dropout(p=drop_rate)
        self.linear = Conv1D(in_dim=visual_dim, out_dim=dim, kernel_size=1, stride=1, bias=True, padding=0)

    def forward(self, visual_features):
        visual_features = self.drop(visual_features)
        output = self.linear(visual_features)
        return output


class FeatureEncoder(nn.Module):
    def __init__(self, dim, num_heads, max_pos_len, kernel_size=7, num_layers=4, drop_rate=0.0):
        super(FeatureEncoder, self).__init__()

        self.conv_block = DepthwiseSeparableConvBlock(
            dim=dim,
            kernel_size=kernel_size,
            drop_rate=drop_rate,
            num_layers=num_layers
        )
        self.attention_block =AttentionExtractor(
            num_head=num_heads,
            num_feature=dim
        )

    def forward(self, x, mask=None):
        features = self.conv_block(x)
        features = self.attention_block(features)
        return features


class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, dim, kernel_size, drop_rate, num_layers=4):
        super(DepthwiseSeparableConvBlock, self).__init__()

        self.depthwise_separable_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, groups=dim,
                          padding=kernel_size // 2, bias=False),
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, bias=True),
                nn.ReLU(),
            ) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        output = x
        for idx, conv_layer in enumerate(self.depthwise_separable_conv):
            residual = output
            output = self.layer_norms[idx](output)
            output = output.transpose(1, 2)
            output = conv_layer(output)
            output = self.dropout(output)
            output = output.transpose(1, 2) + residual
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head=8, num_feature=1024, hidden_dim=1024):
        super().__init__()
        self.num_head = num_head
        self.hidden_dim = hidden_dim

        self.Q = nn.Linear(num_feature, hidden_dim, bias=False)
        self.K = nn.Linear(num_feature, hidden_dim, bias=False)
        self.V = nn.Linear(num_feature, hidden_dim, bias=False)

        # self.d_k = num_feature // num_head
        self.d_k = hidden_dim // num_head
        self.attention = ScaledDotProductAttention(self.d_k)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, num_feature, bias=False),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        batch_size, seq_len, num_feature = x.shape

        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)

        K = K.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)
        Q = Q.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)
        V = V.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)

        y, attn = self.attention(Q, K, V)
        y = y.view(1, self.num_head, seq_len, self.d_k).permute(0, 2, 1, 3).contiguous().view(1, seq_len, self.hidden_dim)

        y = self.fc(y)

        return y, attn


class AttentionExtractor(MultiHeadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *inputs):
        out, attn = super().forward(*inputs)
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()

        self.dropout = nn.Dropout(0.5)
        self.sqrt_d_k = math.sqrt(d_k)

    def forward(self, Q, K, V):
        attn = torch.bmm(Q, K.transpose(2, 1))
        attn = attn / self.sqrt_d_k

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        y = torch.bmm(attn, V)

        return y, attn


class CQAttention(nn.Module):
    def __init__(self, dim, drop_rate=0.0):
        super(CQAttention, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(context, query)
        score_ = nn.Softmax(dim=2)(mask_logits(score, q_mask.unsqueeze(1)))
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))
        score_t = score_t.transpose(1, 2)
        c2q = torch.matmul(score_, query)
        q2c = torch.matmul(torch.matmul(score_, score_t), context)
        output = torch.cat([context, c2q, torch.mul(context, c2q), torch.mul(context, q2c)], dim=2)
        output = self.cqa_linear(output)
        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand([-1, -1, q_seq_len])
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand([-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2
        return res


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


class CQConcatenate(nn.Module):
    def __init__(self, dim):
        super(CQConcatenate, self).__init__()

        self.weighted_pool = WeightedPool(dim=dim)
        self.conv1d = Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, q_mask):
        pooled_query = self.weighted_pool(query, q_mask)
        _, c_seq_len, _ = context.shape
        pooled_query = pooled_query.unsqueeze(1).repeat(1, c_seq_len, 1)
        output = torch.cat([context, pooled_query], dim=2)
        output = self.conv1d(output)
        return output


class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(x, self.weight, dims=1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x


class HighLightLayer(nn.Module):
    def __init__(self, dim):
        super(HighLightLayer, self).__init__()
        self.conv1d = Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, mask):
        logits = self.conv1d(x)
        logits = logits.squeeze(2)
        logits = mask_logits(logits, mask)
        scores = nn.Sigmoid()(logits)
        return scores

    @staticmethod
    def compute_loss(scores, labels, mask, epsilon=1e-12):
        labels = labels.type(torch.float32)
        weights = torch.where(labels == 0.0, labels + 1.0, 2.0 * labels)
        loss_per_location = nn.BCELoss(reduction='none')(scores, labels)
        loss_per_location = loss_per_location * weights
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + epsilon)
        return loss


class ConditionedPredictor(nn.Module):
    def __init__(self, dim):
        super(ConditionedPredictor, self).__init__()
        self.start_rnn = DynamicRNN(dim=dim)
        self.end_rnn = DynamicRNN(dim=dim)

        self.start_block = nn.Sequential(
            Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.end_block = nn.Sequential(
            Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x, mask):
        start_features = self.start_rnn(x, mask)
        end_features = self.end_rnn(start_features, mask)
        start_features = self.start_block(torch.cat([start_features, x], dim=2))
        end_features = self.end_block(torch.cat([end_features, x], dim=2))
        start_logits = mask_logits(start_features.squeeze(2), mask=mask)
        end_logits = mask_logits(end_features.squeeze(2), mask=mask)
        return start_logits, end_logits

    @staticmethod
    def extract_index(start_logits, end_logits):
        start_prob = nn.Softmax(dim=1)(start_logits)
        end_prob = nn.Softmax(dim=1)(end_logits)
        outer = torch.matmul(start_prob.unsqueeze(dim=2), end_prob.unsqueeze(dim=1))
        outer = torch.triu(outer, diagonal=0)
        _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)
        _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)
        return start_index, end_index

    @staticmethod
    def compute_cross_entropy_loss(start_logits, end_logits, start_labels, end_labels):
        start_loss = nn.CrossEntropyLoss(reduction='mean')(start_logits, start_labels)
        end_loss = nn.CrossEntropyLoss(reduction='mean')(end_logits, end_labels)
        return start_loss + end_loss


class DynamicRNN(nn.Module):
    def __init__(self, dim):
        super(DynamicRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, bias=True, batch_first=True,
                            bidirectional=False)

    def forward(self, x, mask):
        out, _ = self.lstm(x)
        mask = mask.type(torch.float32)
        mask = mask.unsqueeze(2)
        out = out * mask
        return out
