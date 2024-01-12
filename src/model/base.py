import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class Base(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.pad_value = tokenizer.convert_token_to_id(tokenizer.pad_token, tokenizer.item_vocab)
        user_vocab_size = len(tokenizer.user_vocab)
        item_vocab_size = len(tokenizer.item_vocab)
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


def base(tokenizer):
    model = Base(tokenizer)
    return model
