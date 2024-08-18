import torch
import torch.nn as nn


class Base(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.register_buffer('base', torch.zeros(num_items, dtype=torch.float32))
        self.register_buffer('count', torch.zeros(num_items, dtype=torch.long))
        self.eps = 1e-8
        self.place_holder = nn.Parameter(torch.zeros(1, ))

    def forward(self, user, item, rating, item_hist):
        if self.training:
            self.base.scatter_add_(0, item.view(-1), rating.view(-1).to(self.base.dtype))
            self.count.scatter_add_(0, item.view(-1), self.count.new_ones(rating.view(-1).size()))
        output_rating = self.base[item] / (self.count[item] + self.eps)
        return output_rating


def base(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    model = Base(num_users, num_items)
    return model
