import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class Base(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.num_items = len(tokenizer.item_vocab)
        self.register_buffer('base', torch.zeros(self.num_items))
        self.register_buffer('count', torch.zeros(self.num_items))
        self.place_holder = nn.Parameter(torch.zeros(1,))

    def forward(self, input):
        output = {}
        print(input)
        exit()
        if cfg['target_mode'] == 'explicit':
            if self.training:
                self.base.scatter_add_(0, input['item'], input['rating'])
                self.count.scatter_add_(0, input['item'], input['rating'].new_ones(input['rating'].size()))
            output['target_rating'] = self.base[input['target_item']] / (self.count[input['target_item']] + 1e-10)
            output['target_rating'][self.count[input['target_item']] == 0] = (self.base[self.count != 0] /
                                                                              self.count[self.count != 0]).mean()
            output['loss'] = loss_fn(output['target_rating'], input['target_rating'])
        elif cfg['target_mode'] == 'implicit':
            if self.training:
                self.base.scatter_add_(0, input['item'], input['rating'])
                self.count = self.count + torch.unique(input['user']).size(0)
            output['target_rating'] = self.base[input['target_item']] / self.count[input['target_item']]
            output['loss'] = loss_fn(output['target_rating'], input['target_rating'])
        else:
            raise ValueError('Not valid target mode')
        return output


def base(tokenizer):
    model = Base(tokenizer)
    return model
