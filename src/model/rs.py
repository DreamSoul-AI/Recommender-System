import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import make_loss_fn


class RecommenderSystem(nn.Module):
    def __init__(self, base, model_name, loss_mode, loss_kwargs, pad_token):
        super().__init__()
        self.base = base
        self.model_name = model_name
        self.loss_mode = loss_mode
        self.loss_fn = make_loss_fn(loss_mode, loss_kwargs['loss_hyperparam'])
        self.pad_token = pad_token

    @property
    def num_users(self):
        return self.base.num_users

    @property
    def num_items(self):
        return self.base.num_items

    def user_embedding(self, input):
        if self.model_name in ['simplex']:
            user_embedding = self.base.user_embedding(input['user'], input['item_hist'])
        else:
            raise NotImplementedError
        return user_embedding

    def item_embedding(self, input):
        if self.model_name in ['simplex']:
            item_embedding = self.base.item_embedding(input['item'])
        else:
            raise NotImplementedError
        return item_embedding

    def forward(self, input):
        output = {}
        user, item, target, item_hist = input['user'], input['item'], input['target'], input['item_hist']
        if self.model_name not in ['base']:
            pred, user_embedding, item_embedding = self.base(user, item, target, item_hist)
            output['user_embedding'] = user_embedding
            output['item_embedding'] = item_embedding[:, 0]
        else:
            pred = self.base(user, item, target, item_hist)
            output['user_embedding'] = None
            output['item_embedding'] = None
        output['pred'] = pred
        output['loss'] = self.loss_fn(output['pred'], target)
        return output


def rs(model, cfg):
    model_name = cfg['model_name']
    loss_mode = cfg['loss_mode']
    loss_kwargs = cfg['loss_kwargs']
    pad_token = cfg['pad_token']
    model = RecommenderSystem(model, model_name, loss_mode, loss_kwargs, pad_token)
    return model


def normalize_embedding(embedding, score_mode, embedding_side):
    if score_mode == 'cosine':
        embedding = F.normalize(embedding, dim=-1)
    elif score_mode == 'userproj' and embedding_side == 'user':
        embedding = F.normalize(embedding, dim=-1)
    elif score_mode == 'itemproj' and embedding_side == 'item':
        embedding = F.normalize(embedding, dim=-1)
    elif score_mode == 'pearson':
        embedding = F.normalize(embedding - embedding.mean(dim=-1, keepdims=True), dim=-1)
    elif score_mode == 'dot':
        pass
    else:
        raise ValueError(f'score_mode {score_mode} not supported')
    return embedding
