import torch
import torch.nn as nn

from .model import init_param
from .rs import normalize_embedding
from .layers import CapsuleNetwork


# https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/matching/mind.py


class MIND(nn.Module):
    def __init__(self, num_users, num_items, embedding_mode, max_length, hidden_size, interest_num):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_mode = embedding_mode
        self.max_length = max_length['user']
        self.hidden_size = hidden_size
        self.interest_num = interest_num
        self.user_weight = nn.Embedding(self.num_users, self.hidden_size)
        self.item_weight = nn.Embedding(self.num_items + 1, self.hidden_size)
        self.capsule = CapsuleNetwork(self.hidden_size, self.max_length, bilinear_type=0,
                                      interest_num=self.interest_num)
        self.convert_user_weight = nn.Parameter(torch.rand(self.hidden_size * 2, self.hidden_size), requires_grad=True)

    def make_padding(self, hist, padding_idx):
        hist = torch.where(hist == -100, hist.new_ones((1,)) * padding_idx, hist)
        return hist

    def user_embedding(self, user, item_hist):
        user_embedding = self.user_weight(user)
        user_embedding = user_embedding.unsqueeze(1)
        user_embedding = user_embedding.expand([user_embedding.shape[0], self.interest_num,
                                                user_embedding.shape[-1]])  # [256, 4, 16]
        mask = item_hist == -100
        item_hist = item_hist.clone()
        item_hist = self.make_padding(item_hist, self.num_items)
        item_hist_embedding = self.item_weight(item_hist)
        multi_interest_emb = self.capsule(item_hist_embedding, mask)  # [256, 4, 16]
        user_embedding = torch.cat([user_embedding, multi_interest_emb], dim=-1)  # [256, 4, 32]
        user_embedding = torch.matmul(user_embedding, self.convert_user_weight)
        user_embedding = normalize_embedding(user_embedding, self.embedding_mode, 'user')  # 256*4*16
        return user_embedding

    def item_embedding(self, item):
        item_embedding = self.item_weight(item)
        item_embedding = normalize_embedding(item_embedding, self.embedding_mode, 'item')
        return item_embedding

    def forward(self, user, item, item_hist):
        user_embedding = self.user_embedding(user, item_hist)
        item_embedding = self.item_embedding(item)

        pos_item_embedding = item_embedding[:, 0, :]
        dot_res = torch.bmm(user_embedding, pos_item_embedding.squeeze(1).unsqueeze(-1))
        k_index = torch.argmax(dot_res, dim=1)
        best_interest_emb = torch.rand(user_embedding.shape[0], user_embedding.shape[2], device=user_embedding.device)
        for k in range(user_embedding.shape[0]):
            best_interest_emb[k, :] = user_embedding[k, k_index[k], :]
        best_interest_emb = best_interest_emb.unsqueeze(1)
        output_rating = torch.mul(best_interest_emb, item_embedding).sum(dim=-1)
        return output_rating, user_embedding, item_embedding


def mind(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    max_length = cfg['max_length']
    model = MIND(num_users, num_items, embedding_mode, max_length, **cfg['mind'])
    model.apply(init_param)
    return model
