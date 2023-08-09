import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn
from config import cfg


class MF(nn.Module):
    def __init__(self, num_users, num_items, hidden_size, info_size):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.info_size = info_size
        self.user_weight = nn.Embedding(num_users, hidden_size)
        self.item_weight = nn.Embedding(num_items, hidden_size)
        # self.user_bias = nn.Embedding(num_users, 1)
        # self.item_bias = nn.Embedding(num_items, 1)
        self.scaler = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))
        # self.bias = nn.Parameter(torch.randn(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_weight.weight, 0.0, 1e-4)
        nn.init.normal_(self.item_weight.weight, 0.0, 1e-4)
        # nn.init.zeros_(self.user_bias.weight)
        # nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.scaler)
        nn.init.zeros_(self.bias)
        return

    def user_embedding(self, user):
        embedding = self.user_weight(user)
        if hasattr(self, 'num_matched') and self.md_mode == 'user':
            embedding[user < self.num_matched] = self.md_weight(user[user < self.num_matched]) + self.md_bias(
                user[user < self.num_matched])
        return embedding

    def item_embedding(self, item):
        embedding = self.item_weight(item)
        if hasattr(self, 'num_matched') and self.md_mode == 'item':
            embedding[item < self.num_matched] = self.md_weight(item[item < self.num_matched]) + self.md_bias(
                item[item < self.num_matched])
        return embedding

    def make_md(self, num_matched, md_mode, weight, bias):
        self.num_matched = num_matched
        self.md_mode = md_mode
        self.md_weight = weight
        self.md_bias = bias
        return

    def forward(self, input):
        output = {}
        if self.training:
            user = input['user']
            item = input['item']
            rating = input['rating']
        else:
            user = input['target_user']
            item = input['target_item']
            rating = input['target_rating']
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        # user_embedding = F.normalize(user_embedding - user_embedding.mean(dim=-1, keepdims=True), dim=-1)
        # item_embedding = F.normalize(item_embedding - item_embedding.mean(dim=-1, keepdims=True), dim=-1)
        user_embedding = F.normalize(user_embedding, dim=-1)
        item_embedding = F.normalize(item_embedding, dim=-1)

        mf = torch.bmm(user_embedding.unsqueeze(1), item_embedding.unsqueeze(-1)).squeeze() * self.scaler + 3
        output['target_rating'] = mf
        output['loss'] = loss_fn(output['target_rating'], rating)
        return output


def mf(num_users=None, num_items=None):
    num_users = cfg['num_users']['data'] if num_users is None else num_users
    num_items = cfg['num_items']['data'] if num_items is None else num_items
    hidden_size = cfg['mf']['hidden_size']
    info_size = cfg['info_size']
    model = MF(num_users, num_items, hidden_size, info_size)
    return model
