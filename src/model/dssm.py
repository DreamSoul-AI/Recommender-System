import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import init_param
from .rs import normalize_embedding
from .layers import MLP, AveragePooling


# https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/matching/dssm.py
# https://github.com/reczoo/RecBox/blob/main/recbox/third_party/rechub/models/matching/dssm_facebook.py

class DSSM(nn.Module):
    def __init__(self, num_users, num_items, embedding_mode, hidden_size):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_mode = embedding_mode
        self.hidden_size = hidden_size
        self.user_weight = nn.Embedding(self.num_users, self.hidden_size[-1])
        self.item_weight = nn.Embedding(self.num_items + 1, self.hidden_size[-1])  # +1 for padding index
        self.pooling_layer = AveragePooling()
        self.user_mlp = MLP(hidden_size[-1] * 2, output_layer=False, dims=hidden_size)
        self.item_mlp = MLP(hidden_size[-1], output_layer=False, dims=hidden_size)

    def make_padding(self, hist, padding_idx):
        return torch.where(hist == -100, hist.new_full((1,), padding_idx), hist)

    def user_embedding(self, user, item_hist):
        mask = item_hist == -100
        item_hist = self.make_padding(item_hist, self.num_items)
        item_hist_embedding = self.item_weight(item_hist)
        item_hist_embedding = self.pooling_layer(item_hist_embedding, mask)
        user_embedding = self.user_weight(user)
        user_embedding = torch.cat([user_embedding, item_hist_embedding], dim=-1)
        user_embedding = self.user_mlp(user_embedding)
        user_embedding = normalize_embedding(user_embedding, self.embedding_mode, 'user')
        return user_embedding

    def item_embedding(self, item):
        item_embedding = self.item_weight(item)
        item_embedding_shape = item_embedding.size()
        item_embedding = item_embedding.view(-1, item_embedding.size(-1))
        item_embedding = self.item_mlp(item_embedding)
        item_embedding = item_embedding.view(item_embedding_shape)
        item_embedding = normalize_embedding(item_embedding, self.embedding_mode, 'item')
        return item_embedding

    def forward(self, user, item, item_hist):
        user_embedding = self.user_embedding(user, item_hist)
        item_embedding = self.item_embedding(item)
        if item_embedding.dim() == 2:
            item_embedding = item_embedding.unsqueeze(1)
        output_rating = torch.bmm(item_embedding, user_embedding.unsqueeze(-1)).squeeze(-1)
        return output_rating, user_embedding, item_embedding


def dssm(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    model = DSSM(num_users, num_items, embedding_mode, **cfg['dssm'])
    model.apply(init_param)
    return model
