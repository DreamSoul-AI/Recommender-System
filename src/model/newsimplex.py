import torch
import torch.nn as nn
import torch.nn.functional as F
from .rs import normalize_embedding
from .model import init_param


# https://github.com/reczoo/RecZoo/blob/main/matching/cf/SimpleX/src/SimpleX.py

class NewSimpleX(nn.Module):
    def __init__(self, num_users, num_items, embedding_mode, hidden_size=64, aggregation_mode='mean', gamma=0.5,
                 attention_dropout=0., net_dropout=0., enable_bias=True):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_mode = embedding_mode
        self.hidden_size = hidden_size
        self.aggregation_mode = aggregation_mode
        self.user_weight = nn.Embedding(self.num_users, self.hidden_size)
        self.item_weight = nn.Embedding(self.num_items + 1, self.hidden_size, padding_idx=self.num_items)
        self.behavior_aggregation = BehaviorAggregator(hidden_size, aggregator=aggregation_mode, gamma=gamma,
                                                       dropout_rate=attention_dropout)
        self.dropout = nn.Dropout(p=net_dropout)
        self.enable_bias = enable_bias
        if self.enable_bias:
            self.user_bias = nn.Embedding(self.num_users, 1)
            self.item_bias = nn.Embedding(self.num_items + 1, 1, padding_idx=self.num_items)
            self.global_bias = nn.Parameter(torch.zeros(1, ))

    def make_padding(self, hist, padding_idx):
        hist = torch.where(hist == -100, hist.new_ones((1,)) * padding_idx, hist)
        return hist

    def user_embedding(self, user, item_hist):
        user_embedding = self.user_weight(user)
        item_hist = self.make_padding(item_hist, self.num_items)
        item_hist_embedding = self.item_weight(item_hist)
        user_embedding = self.behavior_aggregation(user_embedding, item_hist_embedding)
        user_embedding = normalize_embedding(user_embedding, self.embedding_mode, 'user')
        if self.enable_bias:
            user_embedding = torch.cat([user_embedding, user_embedding.new_ones(user_embedding.size(0), 1)], dim=-1)
        return user_embedding

    # def item_embedding(self, item):
    #     item_embedding = self.item_weight(item)
    #     item_embedding = normalize_embedding(item_embedding, self.embedding_mode, 'item')
    #     if self.enable_bias:
    #         item_embedding = torch.cat([item_embedding, self.item_bias(item)], dim=-1)
    #     return item_embedding

    def item_embedding(self, user_hist, item):
        item_embedding = self.item_weight(item)

        user_hist = self.make_padding(user_hist, self.num_users)
        user_seq_emb = self.user_weight(user_hist)
        # pick the first positive item
        item_emb = self.behavior_aggregation(item_embedding, user_seq_emb)
        item_emb = normalize_embedding(item_emb, self.embedding_mode, 'item')
        if self.enable_bias:
            item_emb = torch.cat([item_emb, self.item_bias(item)], dim=-1)
        return item_emb

    def forward(self, user, user_hist, item, item_hist):
        user_embedding = self.user_embedding(user, item_hist)
        user_embedding = self.dropout(user_embedding)
        item_embedding = self.item_embedding(user_hist, item)
        if item_embedding.dim() == 2:
            item_embedding = item_embedding.unsqueeze(1)
        output_rating = torch.bmm(item_embedding, user_embedding.unsqueeze(-1)).squeeze(-1)
        if self.enable_bias:  # user_bias and global_bias only influence training, but not inference for ranking
            output_rating = output_rating + self.user_bias(user) + self.global_bias
        return output_rating, user_embedding, item_embedding


class BehaviorAggregator(nn.Module):
    def __init__(self, embedding_dim, gamma=0.5, aggregator="mean", dropout_rate=0.):
        super(BehaviorAggregator, self).__init__()
        self.aggregator = aggregator
        self.gamma = gamma
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        if self.aggregator in ["user_attention", "self_attention"]:
            self.W_k = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                     nn.Tanh())
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
            if self.aggregator == "self_attention":
                self.W_q = nn.Parameter(torch.Tensor(embedding_dim, 1))
                nn.init.xavier_normal_(self.W_q)

    def forward(self, uid_emb, sequence_emb):
        out = uid_emb
        if self.aggregator == "mean":
            out = self.average_pooling(sequence_emb)
        elif self.aggregator == "user_attention":
            out = self.user_attention(uid_emb, sequence_emb)
        elif self.aggregator == "self_attention":
            out = self.self_attention(sequence_emb)
        return self.gamma * uid_emb + (1 - self.gamma) * out

    def user_attention(self, uid_emb, sequence_emb):
        single_query = (uid_emb.dim() == 2)
        if single_query:
            uid_q = uid_emb.unsqueeze(1).  #[B,1,D]
        else:
            uid_q = uid_emb
        key = self.W_k(sequence_emb)  # b x seq_len x attention_dim
        mask = (sequence_emb.sum(dim=-1) == 0)

        logits = torch.matmul(uid_q, key.transpose(1,2))
        logits = logits / (uid_q.size(-1) **0.5) 
        logits = logits.masked_fill(mask.unsqueeze(1), -1e9)
        attention = F.softmax(logits, dim = -1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        #output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        output = torch.matmul(attention, sequence_emb)
        output = self.W_v(output)

        if single_query:
            output = output.squeeze(1)
        return output

    def self_attention(self, sequence_emb):
        key = self.W_k(sequence_emb)  # b x seq_len x attention_dim
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.matmul(key, self.W_q).squeeze(-1)  # b x seq_len
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_v(output)

    def average_pooling(self, sequence_emb):
        mask = sequence_emb.sum(dim=-1) != 0
        mean = sequence_emb.sum(dim=1) / (mask.float().sum(dim=-1, keepdim=True) + 1.e-9)
        return self.W_v(mean)

    def masked_softmax(self, X, mask):
        # use the following softmax to avoid nans when a sequence is entirely masked
        X = X.masked_fill_(mask, 0)
        e_X = torch.exp(X)
        return e_X / (e_X.sum(dim=1, keepdim=True) + 1.e-9)


def newsimplex(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    model = NewSimpleX(num_users, num_items, embedding_mode, **cfg['new_simplex'])
    model.apply(init_param)
    return model
