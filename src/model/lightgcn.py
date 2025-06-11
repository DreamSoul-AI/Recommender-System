import torch
import torch.nn as nn
import torch.nn.functional as F
from .rs import normalize_embedding
from .model import init_param


# https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/general_recommender/lightgcn.py
# https://github.com/recsys-benchmark/DaisyRec-v2.0/blob/dev/daisy/model/LightGCNRecommender.py


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, norm_adj, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.norm_adj = norm_adj  # torch.sparse.FloatTensor
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.final_user_emb = None
        self.final_item_emb = None

    def propagate(self):
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]

        x = ego_embeddings
        for _ in range(self.num_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            all_embeddings.append(x)

        final_embedding = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)
        self.final_user_emb, self.final_item_emb = torch.split(final_embedding, [self.num_users, self.num_items])

    def user_embedding_out(self, user_ids):
        if self.final_user_emb is None:
            self.propagate()
        return self.final_user_emb[user_ids]

    def item_embedding_out(self, item_ids):
        if self.final_item_emb is None:
            self.propagate()
        return self.final_item_emb[item_ids]

    def score(self, user_ids, item_ids):
        u = self.user_embedding_out(user_ids)
        i = self.item_embedding_out(item_ids)
        return (u * i).sum(dim=-1)

    def full_score(self, user_ids):
        u = self.user_embedding_out(user_ids)         # [B, D]
        i = self.final_item_emb                       # [I, D]
        return torch.matmul(u, i.T)                   # [B, I]


def lightgcn(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    model = LightGCN(num_users, num_items, embedding_mode, **cfg['simplex'])
    model.apply(init_param)
    return model
