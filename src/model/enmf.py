import torch
import torch.nn as nn
import torch.nn.functional as F
from .rs import normalize_embedding
from .model import init_param


# https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/general_recommender/enmf.py


class ENMF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, history_matrix, dropout=0.2):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.history_matrix = history_matrix  # shape: [num_users, max_len] on device

        self.user_embedding = nn.Embedding(num_users, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.H_i = nn.Linear(embed_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def user_embedding_out(self, user_ids):
        return self.user_embedding(user_ids)

    def item_embedding_out(self, item_ids):
        return self.item_embedding(item_ids)

    def forward(self, user_ids):
        u_emb = self.user_embedding(user_ids)  # [B, D]
        u_emb = self.dropout(u_emb)  # [B, D]
        history_items = self.history_matrix[user_ids]  # [B, max_len]
        i_emb = self.item_embedding(history_items)  # [B, max_len, D]
        scores = self.H_i(u_emb.unsqueeze(1) * i_emb)  # [B, max_len, 1]
        return scores.squeeze(-1)  # [B, max_len]

    def score(self, user_ids, item_ids):
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)
        return self.H_i(u * i).squeeze(-1)

    def full_score(self, user_ids):
        u = self.user_embedding(user_ids)  # [B, D]
        i_all = self.item_embedding.weight  # [I, D]
        scores = self.H_i(u.unsqueeze(1) * i_all.unsqueeze(0))  # [B, I, 1]
        return scores.squeeze(-1)  # [B, I]


def enmf(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    model = ENMF(num_users, num_items, embedding_mode, **cfg['simplex'])
    model.apply(init_param)
    return model
