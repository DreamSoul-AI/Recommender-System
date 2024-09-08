import torch
import torch.nn as nn
import torch.nn.functional as F


class MF(nn.Module):
    def __init__(self, num_users, num_items, hidden_size):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.user_weight = nn.Embedding(self.num_users, self.hidden_size)
        self.item_weight = nn.Embedding(self.num_items, self.hidden_size)
        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_weight.weight, 0.0, 1e-4)
        nn.init.normal_(self.item_weight.weight, 0.0, 1e-4)
        nn.init.constant_(self.user_bias.weight, 0.0)
        nn.init.constant_(self.item_bias.weight, 0.0)
        return

    def user_embedding(self, user):
        embedding = self.user_weight(user) + self.user_bias(user)
        return embedding

    def item_embedding(self, item):
        embedding = self.item_weight(item) + self.item_bias(item)
        return embedding

    def forward(self, user, item, rating, item_hist):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        return user_embedding, item_embedding


def mf(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    hidden_size = cfg['mf']['hidden_size']
    model = MF(num_users, num_items, hidden_size)
    return model
