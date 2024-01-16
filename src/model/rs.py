import torch
import torch.nn as nn
import torch.nn.functional as F
import model


class RecommenderSystem(nn.Module):
    def __init__(self, tokenizer, base, target_mode, stats):
        super().__init__()
        self.tokenizer = tokenizer
        self.base = base
        self.target_mode = target_mode
        self.stats = stats

    def forward(self, input):
        output = {}
        user, item, rating, attention_mask = input['user'], input['item'], input['rating'], input['attention_mask']
        target_item, target_rating, target_attention_mask = (input['target_item'], input['target_rating'],
                                                             input['target_attention_mask'])
        if self.target_mode == 'explicit':
            rating = model.normalize(rating, self.stats['min'], self.stats['max'])
            target_rating = model.normalize(target_rating, self.stats['min'], self.stats['max'])
        output['rating'], output['target_rating'] = self.base(user, item, rating, attention_mask, target_item,
                                                              target_attention_mask)
        output['loss'] = model.make_loss(output['rating'], rating[attention_mask].detach())
        output['target_loss'] = model.make_loss(output['target_rating'], target_rating[target_attention_mask].detach())
        if self.target_mode == 'explicit':
            output['rating'] = model.denormalize(output['rating'], self.stats['min'], self.stats['max'])
            output['target_rating'] = model.denormalize(output['target_rating'], self.stats['min'], self.stats['max'])
        input['rating'] = input['rating'][attention_mask]
        input['target_rating'] = input['target_rating'][target_attention_mask]
        return output


def rs(tokenizer, model, cfg):
    target_mode = cfg['target_mode']
    stats = cfg['stats']
    model = RecommenderSystem(tokenizer, model, target_mode, stats)
    return model
