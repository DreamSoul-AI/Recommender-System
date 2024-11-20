import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleX(nn.Module):
    def __init__(self, num_users, num_items, hidden_size, aggregation_mode='mean', gamma=0.5,
                 attention_dropout=0., net_dropout=0.):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.aggregation_mode = aggregation_mode
        self.user_weight = nn.Embedding(self.num_users, self.hidden_size)
        self.item_weight = nn.Embedding(self.num_items + 1, self.hidden_size)
        self.behavior_aggregation = BehaviorAggregator(hidden_size, aggregator=aggregation_mode, gamma=gamma,
                                                       dropout_rate=attention_dropout)
        self.dropout = nn.Dropout(p=net_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_weight.weight, 0.0, 1e-4)
        nn.init.normal_(self.item_weight.weight, 0.0, 1e-4)
        return

    def user_embedding(self, user, item_hist):
        user_embedding = self.user_weight(user)
        mask = item_hist == -100
        item_hist[mask] = self.num_items
        item_hist_embedding = self.item_embedding(item_hist)
        item_hist_embedding[mask] = 0
        user_embedding = self.behavior_aggregation(user_embedding, item_hist_embedding)
        embedding = self.dropout(user_embedding)
        return embedding

    def item_embedding(self, item):
        embedding = self.item_weight(item)
        return embedding

    def forward(self, user, item, rating, item_hist):
        user_embedding = self.user_embedding(user, item_hist)
        item_embedding = self.item_embedding(item)
        return user_embedding, item_embedding


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

    def average_pooling(self, sequence_emb):
        mask = sequence_emb.sum(dim=-1) != 0
        mean = sequence_emb.sum(dim=1) / (mask.float().sum(dim=-1, keepdim=True) + 1.e-9)
        return self.W_v(mean)

    def user_attention(self, uid_emb, sequence_emb):
        key = self.W_k(sequence_emb)  # b x seq_len x attention_dim
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.bmm(key, uid_emb.unsqueeze(-1)).squeeze(-1)  # b x seq_len
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_v(output)

    def self_attention(self, sequence_emb):
        key = self.W_k(sequence_emb)  # b x seq_len x attention_dim
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.matmul(key, self.W_q).squeeze(-1)  # b x seq_len
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_v(output)

    def masked_softmax(self, X, mask):
        # use the following softmax to avoid nans when a sequence is entirely masked
        X = X.masked_fill_(mask, 0)
        e_X = torch.exp(X)
        return e_X / (e_X.sum(dim=1, keepdim=True) + 1.e-9)


def simplex(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    hidden_size = cfg['simplex']['hidden_size']
    aggregation_mode = cfg['simplex']['aggregation_mode']
    gamma = cfg['simplex']['gamma']
    attention_dropout = cfg['simplex']['attention_dropout']
    net_dropout = cfg['simplex']['net_dropout']
    model = SimpleX(num_users, num_items, hidden_size, aggregation_mode, gamma, attention_dropout, net_dropout)
    return model
