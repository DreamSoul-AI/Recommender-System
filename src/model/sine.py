import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import init_param
from .rs import normalize_embedding
from torch import einsum


# https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/matching/sine.py

class SINE(nn.Module):
    def __init__(self, num_users, num_items, embedding_mode, max_length, hidden_size, hidden_att_dim, num_intention,
                 num_concepts, num_heads=1, temperature=1.0):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_mode = embedding_mode
        self.max_length = max_length['user']
        self.hidden_size = hidden_size
        self.hidden_att_dim = hidden_att_dim
        self.num_concepts = num_concepts
        self.num_intention = num_intention
        self.temperature = temperature

        self.user_weight = nn.Embedding(self.num_users, self.hidden_size)
        # Item embedding (item IDs + 1 for padding)
        self.item_weight = nn.Embedding(self.num_items + 1, self.hidden_size)

        self.concept_weight = nn.Embedding(self.num_concepts, self.hidden_size)
        self.position_weight = nn.Embedding(self.max_length, self.hidden_size)

        self.w_1 = torch.nn.Parameter(torch.rand(hidden_size, hidden_att_dim), requires_grad=True)
        self.w_2 = torch.nn.Parameter(torch.rand(hidden_att_dim, num_heads), requires_grad=True)

        self.w_3 = torch.nn.Parameter(torch.rand(hidden_size, hidden_size), requires_grad=True)

        self.w_k1 = torch.nn.Parameter(torch.rand(hidden_size, hidden_att_dim), requires_grad=True)
        self.w_k2 = torch.nn.Parameter(torch.rand(hidden_att_dim, num_intention), requires_grad=True)

        self.w_4 = torch.nn.Parameter(torch.rand(hidden_size, hidden_att_dim), requires_grad=True)
        self.w_5 = torch.nn.Parameter(torch.rand(hidden_att_dim, num_heads), requires_grad=True)

    def make_padding(self, hist, padding_idx):
        hist = torch.where(hist == -100, hist.new_ones((1,)) * padding_idx, hist)
        return hist

    def user_embedding(self, user, item_hist):
        # sparse interests extraction
        ## user specific historical item embedding X^u
        mask = item_hist == -100
        item_hist = item_hist.clone()
        item_hist = self.make_padding(item_hist, self.num_items)

        x_u = self.item_weight(item_hist) + self.position_weight.weight.unsqueeze(0)

        x_u_mask = mask.long()

        ## user specific conceptual prototypes C^u
        ### attention a
        h_1 = torch.tanh(einsum("bse,eh->bsh", x_u, self.w_1))
        a_hist = F.softmax(einsum('bsd, dh -> bsh', h_1, self.w_2) + -1.e9 * (1 - x_u_mask.unsqueeze(-1).float()),
                           dim=1)
        ### virtual concept vector z_u
        z_u = einsum("bse, bsh -> be", x_u, a_hist)

        ### similarity between user's concept vector and entire conceptual prototypes s^u
        s_u = einsum("be, te -> bt", z_u, self.concept_weight.weight)
        s_u_top_k = torch.topk(s_u, self.num_intention)

        ### final C^u
        c_u = einsum("bk, bke -> bke", torch.sigmoid(s_u_top_k.values), self.concept_weight(s_u_top_k.indices))

        ## user intention assignment P_{k|t}        
        p_u = F.softmax(einsum("bse, bke -> bks", F.normalize(x_u @ self.w_3, dim=-1), F.normalize(c_u, p=2, dim=-1)),
                        dim=1)

        ## attention weighing P_{t|k}
        h_2 = einsum('bse, ed -> bsd', x_u, self.w_k1).tanh()
        a_concept_k = F.softmax(einsum('bsd, dk -> bsk', h_2, self.w_k2) + -1.e9 * (1 - x_u_mask.unsqueeze(-1).float()),
                                dim=1)

        ## multiple interests encoding \phi_\theta^k(x^u)
        phi_u = einsum("bks, bse -> bke", p_u * a_concept_k.permute(0, 2, 1), x_u)

        # adaptive interest aggregation
        ## intention aware input behavior \hat{X^u}
        x_u_hat = einsum('bks, bke -> bse', p_u, c_u)

        ## user's next intention C^u_{apt}
        h_3 = einsum('bse, ed -> bsd', x_u_hat, self.w_4).tanh()
        c_u_apt = F.normalize(
            einsum(
                "bs, bse -> be",
                F.softmax(einsum('bsd, dh -> bsh', h_3, self.w_5).reshape(-1, self.max_length) + -1.e9 * (
                        1 - x_u_mask.float()), dim=1),
                x_u_hat
            ), -1)

        ## aggregation weights e_k^u
        e_u = F.softmax(einsum('be, bke -> bk', c_u_apt, phi_u) / self.temperature, dim=1)

        # final user representation v^u
        v_u = einsum('bk, bke -> be', e_u, phi_u)
        user_embedding = v_u
        user_embedding = normalize_embedding(user_embedding, self.embedding_mode, 'user')
        return user_embedding

    def item_embedding(self, item):
        item_embedding = self.item_weight(item)
        item_embedding = normalize_embedding(item_embedding, self.embedding_mode, 'item')
        return item_embedding

    def forward(self, user, item, item_hist):
        user_embedding = self.user_embedding(user, item_hist)
        item_embedding = self.item_embedding(item)
        output_rating = torch.bmm(item_embedding, user_embedding.unsqueeze(-1)).squeeze(-1)
        return output_rating, user_embedding, item_embedding


def sine(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    max_length = cfg['max_length']
    model = SINE(num_users, num_items, embedding_mode, max_length, **cfg['sine'])
    model.apply(init_param)
    return model
