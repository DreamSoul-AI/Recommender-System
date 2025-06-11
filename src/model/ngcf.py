import torch
import torch.nn as nn
import torch.nn.functional as F
from .rs import normalize_embedding
from .model import init_param

# https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/general_recommender/ngcf.py
# https://github.com/recsys-benchmark/DaisyRec-v2.0/blob/dev/daisy/model/NGCFRecommender.py

class BiGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.interact_transform = nn.Linear(in_dim, out_dim)

    def forward(self, lap, x):
        ax = torch.sparse.mm(lap, x)
        out1 = self.linear(x + ax)
        out2 = self.interact_transform(x * ax)
        return out1 + out2


class NGCF(nn.Module):
    def __init__(self, num_users, num_items, norm_adj_matrix, embed_dim=64, hidden_sizes=[64, 64],
                 node_dropout=0.1, msg_dropout=0.1):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.norm_adj_matrix = norm_adj_matrix
        self.node_dropout = node_dropout
        self.msg_dropout = msg_dropout

        self.embedding = nn.Embedding(num_users + num_items, embed_dim)
        nn.init.xavier_normal_(self.embedding.weight)

        self.gnn_layers = nn.ModuleList()
        sizes = [embed_dim] + hidden_sizes
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            self.gnn_layers.append(BiGNNLayer(in_dim, out_dim))

        self.final_user_embeddings = None
        self.final_item_embeddings = None

    def sparse_dropout(self, x, dropout_rate):
        if dropout_rate == 0 or not self.training:
            return x
        mask = (torch.rand(x._values().size()) + (1 - dropout_rate)).floor().bool()
        indices = x._indices()[:, mask]
        values = x._values()[mask] * (1.0 / (1 - dropout_rate))
        return torch.sparse.FloatTensor(indices, values, x.size())

    def propagate(self):
        lap = self.sparse_dropout(self.norm_adj_matrix, self.node_dropout)
        x = self.embedding.weight
        out = [x]
        for layer in self.gnn_layers:
            x = layer(lap, x)
            x = F.leaky_relu(x, negative_slope=0.2)
            x = F.dropout(x, p=self.msg_dropout, training=self.training)
            x = F.normalize(x, p=2, dim=-1)
            out.append(x)
        return torch.cat(out, dim=1)

    def forward(self):
        all_embeddings = self.propagate()
        self.final_user_embeddings = all_embeddings[:self.num_users]
        self.final_item_embeddings = all_embeddings[self.num_users:]

    def user_embedding(self, user_ids):
        if self.final_user_embeddings is None:
            self.forward()
        return self.final_user_embeddings[user_ids]

    def item_embedding(self, item_ids):
        if self.final_item_embeddings is None:
            self.forward()
        return self.final_item_embeddings[item_ids]

    def score(self, user_ids, item_ids):
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)
        return (u * i).sum(dim=-1)

    def full_score(self, user_ids):
        u = self.user_embedding(user_ids)  # [B, D]
        v = self.final_item_embeddings  # [I, D]
        return torch.matmul(u, v.T)  # [B, I]


def ngcf(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    model = NGCF(num_users, num_items, embedding_mode, **cfg['simplex'])
    model.apply(init_param)
    return model
