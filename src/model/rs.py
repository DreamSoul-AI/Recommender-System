import torch
import torch.nn as nn
import torch.nn.functional as F
import model
from config import cfg


class RecommenderSystem(nn.Module):
    def __init__(self, tokenizer, base):
        super().__init__()
        self.tokenizer = tokenizer
        self.base = base

    def forward(self, input):
        output = {}
        user, item, rating, attention_mask = input['user'], input['item'], input['rating'], input['attention_mask']
        target_item, target_rating, target_attention_mask = (input['target_item'], input['target_rating'],
                                                             input['target_attention_mask'])
        output['rating'], output['target_rating'] = self.base(user, item, rating, attention_mask, target_item,
                                                              target_attention_mask)
        input['rating'], input['target_rating'] = rating[attention_mask], target_rating[target_attention_mask]
        output['loss'] = model.make_loss(output['rating'], input['rating'])
        output['target_loss'] = model.make_loss(output['target_rating'], input['target_rating'])
        return output


def rs(tokenizer, model):
    model = RecommenderSystem(tokenizer, model)
    return model
