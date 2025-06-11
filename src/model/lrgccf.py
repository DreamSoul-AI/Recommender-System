import torch
import torch.nn as nn
import torch.nn.functional as F
from .rs import normalize_embedding
from .model import init_param


# https://github.com/newlei/LR-GCCF/blob/master/code/train_gowalla.py
# https://github.com/reczoo/BARS/blob/main/matching/libs/LR-GCCF/train_gowalla.py


class LRGCCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, user_item_mat, item_user_mat, u_deg, i_deg):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.user_item_mat = user_item_mat  # Sparse tensor: user × item
        self.item_user_mat = item_user_mat  # Sparse tensor: item × user

        self.embed_user = nn.Embedding(num_users, embed_dim)
        self.embed_item = nn.Embedding(num_items, embed_dim)

        nn.init.normal_(self.embed_user.weight, std=0.0001)
        nn.init.normal_(self.embed_item.weight, std=0.0001)

        # Degree normalization tensors
        self.u_deg = torch.tensor(u_deg, dtype=torch.float32).view(-1, 1).cuda()
        self.i_deg = torch.tensor(i_deg, dtype=torch.float32).view(-1, 1).cuda()

        self.u_deg = self.u_deg.expand(-1, embed_dim)
        self.i_deg = self.i_deg.expand(-1, embed_dim)

        self.final_user_emb = None
        self.final_item_emb = None

    def propagate(self):
        e_u = self.embed_user.weight
        e_i = self.embed_item.weight

        u1 = torch.sparse.mm(self.user_item_mat, e_i) + e_u * self.u_deg
        i1 = torch.sparse.mm(self.item_user_mat, e_u) + e_i * self.i_deg

        u2 = torch.sparse.mm(self.user_item_mat, i1) + u1 * self.u_deg
        i2 = torch.sparse.mm(self.item_user_mat, u1) + i1 * self.i_deg

        u3 = torch.sparse.mm(self.user_item_mat, i2) + u2 * self.u_deg
        i3 = torch.sparse.mm(self.item_user_mat, u2) + i2 * self.i_deg

        final_u = torch.cat([e_u, u1, u2, u3], dim=-1)
        final_i = torch.cat([e_i, i1, i2, i3], dim=-1)

        self.final_user_emb = final_u
        self.final_item_emb = final_i

    def user_embedding(self, user_ids):
        if self.final_user_emb is None:
            self.propagate()
        return self.final_user_emb[user_ids]

    def item_embedding(self, item_ids):
        if self.final_item_emb is None:
            self.propagate()
        return self.final_item_emb[item_ids]

    def score(self, user_ids, item_ids):
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)
        return (u * i).sum(dim=-1)

    def full_score(self, user_ids):
        u = self.user_embedding(user_ids)  # [B, D]
        v = self.final_item_emb  # [I, D]
        return torch.matmul(u, v.T)  # [B, I]


def lrgccf(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    model = LRGCCF(num_users, num_items, embedding_mode, **cfg['simplex'])
    model.apply(init_param)
    return model
