import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch import sigmoid
from torch.nn import GRU, Embedding, Dropout, Parameter
from .rs import normalize_embedding
from .model import init_param


# https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/matching/narm.py

class NARM(nn.Module):
    def __init__(self, num_users, num_items, embedding_mode, hidden_size, emb_dropout_p=0.25,
                 session_rep_dropout_p=0.5):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_mode = embedding_mode
        self.hidden_size = hidden_size
        self.embedding_dim = hidden_size

        self.item_emb = Embedding(num_items + 1, hidden_size, padding_idx=num_items)
        self.emb_dropout = Dropout(emb_dropout_p)

        self.gru = GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

        self.a_1 = Parameter(torch.randn(hidden_size, hidden_size))
        self.a_2 = Parameter(torch.randn(hidden_size, hidden_size))
        self.v = Parameter(torch.randn(hidden_size, 1))
        self.b = Parameter(torch.randn(hidden_size, hidden_size * 2))

        self.session_rep_dropout = Dropout(session_rep_dropout_p)

    def make_padding(self, hist, padding_idx):
        return torch.where(hist == -100, hist.new_full((1,), padding_idx), hist)

    def user_embedding(self, item_hist):
        item_hist = self.make_padding(item_hist, self.num_items)  # (B, L)
        value_mask = (item_hist != self.num_items)  # (B, L)
        lengths = value_mask.sum(dim=1).cpu()

        item_emb_seq = self.item_emb(item_hist)  # (B, L, D)
        packed_embs = rnn_utils.pack_padded_sequence(self.emb_dropout(item_emb_seq), lengths, batch_first=True,
                                                     enforce_sorted=False)

        h_seq, h_last = self.gru(packed_embs)
        h_last = h_last.permute(1, 0, 2)  # (B, 1, H)
        h_seq, _ = rnn_utils.pad_packed_sequence(h_seq, batch_first=True)  # (B, L1, H)

        value_mask = value_mask[:, :h_seq.size(1)]  # Align mask shape with h_seq

        c_g = h_last.squeeze(1)  # (B, H)

        q = sigmoid(h_last @ self.a_1.T + h_seq @ self.a_2.T) @ self.v  # (B, L1, 1)
        alpha = torch.exp(q) * value_mask.unsqueeze(-1)  # (B, L1, 1)
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-9)

        c_l = (alpha * h_seq).sum(1)  # (B, H)
        c = self.session_rep_dropout(torch.cat([c_g, c_l], dim=-1))  # (B, 2H)

        user_emb = c @ self.b.T  # (B, D)
        user_emb = normalize_embedding(user_emb, self.embedding_mode, 'user')
        return user_emb

    def item_embedding(self, item):
        item_emb = self.item_emb(item)  # (B, D) or (N, D)
        item_emb = normalize_embedding(item_emb, self.embedding_mode, 'item')
        return item_emb

    def forward(self, user, item, item_hist):
        user_emb = self.user_embedding(item_hist)  # (B, D)
        item_emb = self.item_embedding(item)  # (B, D) or (B, 1, D)
        if item_emb.dim() == 2:
            item_emb = item_emb.unsqueeze(1)  # (B, 1, D)
        score = torch.bmm(item_emb, user_emb.unsqueeze(-1)).squeeze(-1)
        return score, user_emb, item_emb


def narm(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    model = NARM(num_users, num_items, embedding_mode, **cfg['narm'])
    model.apply(init_param)
    return model
