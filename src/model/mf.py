import torch
import torch.nn as nn
import torch.nn.functional as F
from .rs import normalize_embedding
from .model import init_param


class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_mode, hidden_size=64, enable_bias=True):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_mode = embedding_mode
        self.hidden_size = hidden_size
        self.user_weight = nn.Embedding(self.num_users, self.hidden_size)
        self.item_weight = nn.Embedding(self.num_items, self.hidden_size)
        # self.item_weight = nn.Embedding(self.num_items + 1, self.hidden_size, padding_idx=self.num_items)
        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)

        self.enable_bias = enable_bias
        if self.enable_bias:
            self.user_bias = nn.Embedding(self.num_users, 1)
            self.item_bias = nn.Embedding(self.num_items, 1)
            # self.item_bias = nn.Embedding(self.num_items + 1, 1, padding_idx=self.num_items)
            self.global_bias = nn.Parameter(torch.zeros(1, ))

    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     nn.init.normal_(self.user_weight.weight, 0.0, 1e-4)
    #     nn.init.normal_(self.item_weight.weight, 0.0, 1e-4)
    #     nn.init.constant_(self.user_bias.weight, 0.0)
    #     nn.init.constant_(self.item_bias.weight, 0.0)
    #     return

    def user_embedding(self, user):
        user_embedding = self.user_weight(user) + self.user_bias(user)
        user_embedding = normalize_embedding(user_embedding, self.embedding_mode, 'user')
        if self.enable_bias:
            user_embedding = torch.cat([user_embedding, user_embedding.new_ones(user_embedding.size(0), 1)], dim=-1)
        return user_embedding

    def item_embedding(self, item):
        item_embedding = self.item_weight(item) + self.item_bias(item)
        item_embedding = normalize_embedding(item_embedding, self.embedding_mode, 'item')
        if self.enable_bias:
            item_embedding = torch.cat([item_embedding, self.item_bias(item)], dim=-1)
        return item_embedding

    def forward(self, user, item):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        if item_embedding.dim() == 2:
            item_embedding = item_embedding.unsqueeze(1)
        output_rating = torch.bmm(item_embedding, user_embedding.unsqueeze(-1)).squeeze(-1)
        if self.enable_bias:  # user_bias and global_bias only influence training, but not inference for ranking
            output_rating = output_rating + self.user_bias(user) + self.global_bias
        return output_rating, user_embedding, item_embedding


def mf(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    model = MF(num_users, num_items, embedding_mode,  **cfg['mf'])
    model.apply(init_param)
    return model
