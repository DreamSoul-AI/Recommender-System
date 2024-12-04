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
        # self.enable_bias = enable_bias
        # if self.enable_bias:
        #     self.user_bias = nn.Embedding(self.num_users, 1)
        #     self.item_bias = nn.Embedding(self.num_items + 1, 1)

    # def reset_parameters(self):
    #     nn.init.normal_(self.global_weight, 0.0, 1e-4)
    #     if self.enable_bias:
    #         nn.init.constant_(self.user_bias.weight, 0.0)
    #         nn.init.constant_(self.item_bias.weight, 0.0)
    #         nn.init.constant_(self.global_bias, 0.0)
    #     return

    def forward(self, input):
        output = {}
        user, item, target, item_hist = input['user'], input['item'], input['target'], input['item_hist']
        if self.model_name not in ['base']:
            pred, user_embedding, item_embedding = self.base(user, item, target, item_hist)
            # pred, user_embedding, item_embedding = self.make_score(user_embedding, item_embedding)
            # if self.enable_bias:
            #     user_embedding = torch.cat([user_embedding, user_embedding.new_ones(user_embedding.size(0), 1)], dim=-1)
            #     item_embedding = torch.cat([item_embedding, self.item_bias(item)], dim=-1)
            output['user_embedding'] = user_embedding
            output['item_embedding'] = item_embedding[:, 0]
        else:
            pred = self.base(user, item, target, item_hist)
            output['user_embedding'] = None
            output['item_embedding'] = None
        # pred = self.global_weight * pred
        # if self.enable_bias:
        #     pred += self.user_bias(user) + self.global_bias
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
