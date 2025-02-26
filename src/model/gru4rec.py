import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import init_param
from .rs import normalize_embedding
from .layers import MLP


# https://github.com/reczoo/RecBox/blob/main/recbox/third_party/rechub/models/matching/gru4rec.py

class GRU4Rec(nn.Module):
    def __init__(self, num_users, num_items, embedding_mode, hidden_size, num_layers=2):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_mode = embedding_mode
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.padding_idx = num_items  # Padding index is set to the last index
        self.user_weight = nn.Embedding(self.num_users, hidden_size[-1])
        self.item_weight = nn.Embedding(self.num_items + 1, hidden_size[-1])
        self.user_mlp = MLP(hidden_size[-1] * 2, output_layer=False, dims=hidden_size)
        # GRU layer for modeling user behavior sequences
        self.gru = nn.GRU(
            input_size=hidden_size[-1],
            hidden_size=hidden_size[-1],
            num_layers=num_layers,
            batch_first=True,
            bias=False
        )
        # Embedding layers for users and items
        # self.user_embedding_layer = nn.Embedding(num_users, hidden_size)
        # self.item_embedding_layer = nn.Embedding(
        #     self.num_items + 1, hidden_size)

        # self.user_mlp = nn.Sequential(
        #     nn.Linear(self.hidden_size * 2, self.hidden_size),
        #     nn.ReLU()
        # )



        # self.reset_parameters()

    # def reset_parameters(self):
    #     # Initialize embeddings
    #     nn.init.normal_(self.user_embedding_layer.weight, 0.0, 1e-4)
    #     nn.init.normal_(self.item_embedding_layer.weight, 0.0, 1e-4)
    #     # Ensure the padding index in item embeddings is zero
    #     with torch.no_grad():
    #         self.item_embedding_layer.weight[self.padding_idx].fill_(0)

    def make_padding(self, hist, padding_idx):
        hist = torch.where(hist == -100, hist.new_ones((1,)) * padding_idx, hist)
        return hist

    def user_embedding(self, user, item_hist):
        # item_history = item_hist.clone()
        # item_history[item_history == -100] = self.padding_idx
        # user_emb = self.user_embedding_layer(user)  # Shape: [batch_size, hidden_size]
        # item_hist_emb = self.item_embedding_layer(item_history)  # Shape: [batch_size, seq_len, hidden_size]
        # lengths = (item_history != self.padding_idx).sum(dim=1)  # Shape: [batch_size]

        item_hist = self.make_padding(item_hist, self.num_items)
        lengths = (item_hist != self.num_items).sum(dim=1).tolist()
        item_hist_embedding = self.item_weight(item_hist)
        # print(item_hist_embedding.size())
        # print(lengths)
        item_hist_embedding = nn.utils.rnn.pack_padded_sequence(
            item_hist_embedding, lengths, batch_first=True, enforce_sorted=False
        )
        # print(item_hist_embedding)

        _, item_hist_embedding = self.gru(item_hist_embedding)
        # print(item_hist_embedding.size())
        # exit()

        # hidden shape: [num_layers, batch_size, hidden_size]
        # Use the last layer's hidden state
        item_hist_embedding = item_hist_embedding[-1]  # get last layer hidden, Shape: [batch_size, hidden_size]
        # user_embedding = torch.cat([user_emb, hidden], dim=-1)
        user_embedding = self.user_weight(user)
        user_embedding = torch.cat([user_embedding, item_hist_embedding], dim=-1)
        user_embedding = self.user_mlp(user_embedding)
        user_embedding = normalize_embedding(user_embedding, self.embedding_mode, 'user')
        return user_embedding

    def item_embedding(self, item):
        item_embedding = self.item_weight(item)
        item_embedding = normalize_embedding(item_embedding, self.embedding_mode, 'item')
        return item_embedding

    def forward(self, user, item, item_hist):
        user_embedding = self.user_embedding(user, item_hist)
        item_embedding = self.item_embedding(item)
        if item_embedding.dim() == 2:
            item_embedding = item_embedding.unsqueeze(1)
        output_rating = torch.bmm(item_embedding, user_embedding.unsqueeze(-1)).squeeze(-1)
        return output_rating, user_embedding, item_embedding


def gru4rec(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    model = GRU4Rec(num_users, num_items, embedding_mode, **cfg['youtubednn'])
    model.apply(init_param)
    return model
