import torch
import torch.nn as nn
import torch.nn.functional as F
from .rs import normalize_embedding
from .model import init_param


class STAMP(nn.Module):
    def __init__(self, num_users, num_items, embedding_mode, hidden_size=64, weight_std=0.01, emb_std=0.01, net_dropout=0.0):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_mode = embedding_mode
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=net_dropout)

        # Item embedding (+1 for padding)
        self.item_weight = nn.Embedding(self.num_items + 1, self.hidden_size, padding_idx=self.num_items)

        # STAMP attention parameters
        self.w_0 = nn.Parameter(torch.zeros(self.hidden_size, 1))
        self.w_1_t = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        self.w_2_t = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        self.w_3_t = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        self.b_a = nn.Parameter(torch.zeros(self.hidden_size))

        # Output MLPs
        self.f_s = nn.Sequential(nn.Tanh(), nn.Linear(self.hidden_size, self.hidden_size))
        self.f_t = nn.Sequential(nn.Tanh(), nn.Linear(self.hidden_size, self.hidden_size))

        self._init_params(weight_std, emb_std)

    def _init_params(self, weight_std, emb_std):
        nn.init.normal_(self.item_weight.weight, std=emb_std)
        nn.init.normal_(self.w_0, std=weight_std)
        nn.init.normal_(self.w_1_t, std=weight_std)
        nn.init.normal_(self.w_2_t, std=weight_std)
        nn.init.normal_(self.w_3_t, std=weight_std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=emb_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def make_padding(self, hist, padding_idx):
        return torch.where(hist == -100, hist.new_full((1,), padding_idx), hist)

    def user_embedding(self, item_hist):
        item_hist = self.make_padding(item_hist, self.num_items)                      # (B, L)
        mask = (item_hist != self.num_items).unsqueeze(-1)                            # (B, L, 1)
        seq_len = mask.sum(dim=1, keepdim=True)                                       # (B, 1, 1)

        emb_seq = self.item_weight(item_hist) * mask                                  # (B, L, D)
        last_idx = torch.gather(item_hist, 1, seq_len.squeeze(-1) - 1)                # (B, 1)
        x_t = self.item_weight(last_idx)                                              # (B, 1, D)

        m_s = emb_seq.sum(1) / (seq_len.squeeze(-1) + 1e-9)                           # (B, D)
        m_s = m_s.unsqueeze(1)                                                        # (B, 1, D)

        attention_input = emb_seq @ self.w_1_t + x_t @ self.w_2_t + m_s @ self.w_3_t + self.b_a  # (B, L, D)
        a = torch.exp(torch.sigmoid(attention_input) @ self.w_0) * mask              # (B, L, 1)
        a = F.normalize(a, p=1, dim=1)                                                # (B, L, 1)

        m_a = (a * emb_seq).sum(1) + m_s.squeeze(1)                                   # (B, D)
        h_s = self.f_s(m_a)                                                           # (B, D)
        h_t = self.f_t(x_t).squeeze(1)                                                # (B, D)

        user_emb = h_s * h_t                                                          # (B, D)
        user_emb = normalize_embedding(user_emb, self.embedding_mode, 'user')
        return user_emb

    def item_embedding(self, item):
        item_emb = self.item_weight(item)
        item_emb = normalize_embedding(item_emb, self.embedding_mode, 'item')
        return item_emb

    def forward(self, user, item, item_hist):
        user_emb = self.user_embedding(item_hist)                                     # (B, D)
        user_emb = self.dropout(user_emb)
        item_emb = self.item_embedding(item)                                          # (B, D) or (B, 1, D)
        if item_emb.dim() == 2:
            item_emb = item_emb.unsqueeze(1)
        score = torch.bmm(item_emb, user_emb.unsqueeze(-1)).squeeze(-1)              # (B, 1)
        return score, user_emb, item_emb


def stamp(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    model = STAMP(num_users, num_items, embedding_mode, **cfg['stamp'])
    model.apply(init_param)
    return model
