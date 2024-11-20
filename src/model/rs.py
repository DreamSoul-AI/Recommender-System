import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import make_loss_fn


class RecommenderSystem(nn.Module):
    def __init__(self, base, model_name, score_mode, loss_mode, loss_kwargs, enable_bias=False):
        super().__init__()
        self.base = base
        self.model_name = model_name
        self.score_mode = score_mode
        self.loss_mode = loss_mode
        self.loss_fn = make_loss_fn(loss_mode, loss_kwargs)
        self.global_weight = nn.Parameter(torch.ones(1, ))
        self.global_bias = nn.Parameter(torch.ones(1, ))
        self.pad_token = self.base.num_items
        self.enable_bias = enable_bias
        if self.enable_bias:
            self.user_bias = nn.Embedding(self.num_users, 1)
            self.item_bias = nn.Embedding(self.num_items + 1, 1)

    def reset_parameters(self):
        nn.init.normal_(self.global_weight, 0.0, 1e-4)
        if self.enable_bias:
            nn.init.constant_(self.user_bias.weight, 0.0)
            nn.init.constant_(self.item_bias.weight, 0.0)
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
        elif self.score_mode == 'pearson':
            user_embedding = F.normalize(user_embedding - user_embedding.mean(dim=-1, keepdims=True), dim=-1)
            item_embedding = F.normalize(item_embedding - item_embedding.mean(dim=-1, keepdims=True), dim=-1)
        elif self.score_mode == 'dot':
            pass
        else:
            raise ValueError(f'score_mode {self.score_mode} not supported')
        if item_embedding.dim() == 2:
            item_embedding = item_embedding.unsqueeze(1)
        output_rating = torch.bmm(item_embedding, user_embedding.unsqueeze(-1)).squeeze(-1)
        output_rating = self.global_weight * output_rating + self.global_bias
        return output_rating, user_embedding, item_embedding

    def forward(self, input):
        output = {}
        user, item, target, item_hist = input['user'], input['item'], input['target'], input['item_hist']
        if self.model_name not in ['base']:
            user_embedding, item_embedding = self.base(user, item, target, item_hist)
            pred, user_embedding, item_embedding = self.make_score(user_embedding, item_embedding)
            if self.enable_bias:
                user_embedding = torch.cat([user_embedding, user_embedding.new_ones(user_embedding.size(0), 1)], dim=-1)
                item_embedding = torch.cat([item_embedding, self.item_bias(item)], dim=-1)
            output['user_embedding'] = user_embedding
            output['item_embedding'] = item_embedding[:, 0]
        else:
            pred = self.base(user, item, target, item_hist)
            output['user_embedding'] = None
            output['item_embedding'] = None
        pred = self.global_weight * pred
        if self.enable_bias:
            pred += self.user_bias(user) + self.global_bias
        output['pred'] = pred
        output['loss'] = self.loss_fn(output['pred'], target)
        return output


def rs(model, cfg):
    model_name = cfg['model_name']
    score_mode = cfg['score_mode']
    loss_mode = cfg['loss_mode']
    loss_kwargs = cfg['loss_kwargs']
    enable_bias = cfg['enable_bias']
    model = RecommenderSystem(model, model_name, score_mode, loss_mode, loss_kwargs, enable_bias)
    return model
