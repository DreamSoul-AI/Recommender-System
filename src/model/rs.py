import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import make_loss_fn


class RecommenderSystem(nn.Module):
    def __init__(self, base, model_name, score_mode, loss_mode):
        super().__init__()
        self.base = base
        self.model_name = model_name
        self.score_mode = score_mode
        self.loss_mode = loss_mode
        self.loss_fn = make_loss_fn(loss_mode)
        self.global_weight = nn.Parameter(torch.ones(1, ))
        self.global_bias = nn.Parameter(torch.ones(1, ))

    def reset_parameters(self):
        nn.init.normal_(self.global_weight, 0.0, 1e-4)
        nn.init.constant_(self.global_bias, 0.0)
        return

    def make_score(self, user_embedding, item_embedding):
        if self.score_mode == 'cosine':
            user_embedding = F.normalize(user_embedding, dim=-1)
            item_embedding = F.normalize(item_embedding, dim=-1)
        elif self.score_mode == 'userproj':
            user_embedding = F.normalize(user_embedding, dim=-1)
        elif self.score_mode == 'itemproj':
            item_embedding = F.normalize(item_embedding, dim=-1)
        elif self.score_mode == 'dot':
            pass
        else:
            raise ValueError(f'score_mode {self.score_mode} not supported')
        output_rating = torch.bmm(item_embedding, user_embedding.unsqueeze(-1)).squeeze(-1)
        output_rating = self.global_weight * output_rating + self.global_bias
        return output_rating

    def forward(self, input):
        output = {}
        user, item, rating, item_hist = input['user'], input['item'], input['rating'], input['item_hist']
        if self.model_name not in ['base']:
            user_embedding, item_embedding = self.base(user, item, rating, item_hist)
            output_rating = self.make_score(user_embedding, item_embedding)
        else:
            output_rating = self.base(user, item, rating, item_hist)
        output_rating = self.global_weight * output_rating + self.global_bias
        output['rating'] = output_rating
        output['loss'] = self.loss_fn(output['rating'], rating)
        return output


def rs(model, cfg):
    model_name = cfg['model_name']
    score_mode = cfg['score_mode']
    loss_mode = cfg['loss_mode']
    model = RecommenderSystem(model, model_name, score_mode, loss_mode)
    return model
