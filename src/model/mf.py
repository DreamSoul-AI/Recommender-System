import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MF(nn.Module):
    def __init__(self, user_vocab_size, item_vocab_size, hidden_size):
        super().__init__()
        self.user_vocab_size = user_vocab_size
        self.item_vocab_size = item_vocab_size
        self.hidden_size = hidden_size
        self.user_weight = nn.Embedding(self.user_vocab_size, self.hidden_size)
        self.item_weight = nn.Embedding(self.item_vocab_size, self.hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_weight.weight, 0.0, 1e-4)
        nn.init.normal_(self.item_weight.weight, 0.0, 1e-4)
        return

    def user_embedding(self, user):
        embedding = self.user_weight(user)
        return embedding

    def item_embedding(self, item):
        embedding = self.item_weight(item)
        return embedding

    def forward(self, user, item, rating, attention_mask, target_item, target_attention_mask):
        shape = item.shape
        target_shape = target_item.shape
        target_user = user.expand(target_shape)
        user = user.expand(shape)
        user = user[attention_mask]
        item = item[attention_mask]
        target_user = target_user[target_attention_mask]
        target_item = target_item[target_attention_mask]

        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        target_user_embedding = self.user_embedding(target_user)
        target_item_embedding = self.item_embedding(target_item)
        user_embedding = F.normalize(user_embedding - user_embedding.mean(dim=-1, keepdims=True), dim=-1)
        item_embedding = F.normalize(item_embedding - item_embedding.mean(dim=-1, keepdims=True), dim=-1)

        target_user_embedding = F.normalize(target_user_embedding -
                                            target_user_embedding.mean(dim=-1, keepdims=True), dim=-1)
        target_item_embedding = F.normalize(target_item_embedding -
                                            target_item_embedding.mean(dim=-1, keepdims=True), dim=-1)

        mf = torch.bmm(user_embedding.unsqueeze(-2), item_embedding.unsqueeze(-1))
        target_mf = torch.bmm(target_user_embedding.unsqueeze(-2),
                              target_item_embedding.unsqueeze(-1))
        output_rating = mf.squeeze([-2, -1])
        output_target_rating = target_mf.squeeze([-2, -1])
        return output_rating, output_target_rating


def mf(cfg):
    user_vocab_size = cfg['user_vocab_size']
    item_vocab_size = cfg['item_vocab_size']
    hidden_size = cfg['mf']['hidden_size']
    model = MF(user_vocab_size, item_vocab_size, hidden_size)
    return model
