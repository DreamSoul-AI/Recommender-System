import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class Base(nn.Module):
    def __init__(self, user_vocab_size, item_vocab_size):
        super().__init__()
        self.register_buffer('base', torch.zeros(item_vocab_size))
        self.register_buffer('count', torch.zeros(item_vocab_size))
        self.eps = 1e-8
        self.place_holder = nn.Parameter(torch.zeros(1, ))

    def forward(self, user, item, rating, attention_mask, target_item, target_attention_mask):
        item = item[attention_mask]
        target_item = target_item[target_attention_mask]
        if cfg['target_mode'] == 'explicit':
            if self.training and rating is not None:
                rating = rating[attention_mask]
                self.base.scatter_add_(0, item, rating)
                self.count.scatter_add_(0, item, rating.new_ones(rating.size()))
            output_rating = self.base[item] / (self.count[item] + self.eps)
            output_target_rating = self.base[target_item] / (self.count[target_item] + self.eps)
        elif cfg['target_mode'] == 'implicit':
            if self.training:
                rating = rating[attention_mask]
                self.base.scatter_add_(0, item, rating)
                self.count.scatter_add_(0, item, rating.new_ones(rating.size()))
            output_rating = self.base[item] / (self.count[item] + self.eps)
            output_target_rating = self.base[target_item] / (self.count[target_item] + self.eps)
        else:
            raise ValueError('Not valid target mode')
        return output_rating, output_target_rating


def base(cfg):
    user_vocab_size = cfg['user_vocab_size']
    item_vocab_size = cfg['item_vocab_size']
    model = Base(user_vocab_size, item_vocab_size)
    return model
