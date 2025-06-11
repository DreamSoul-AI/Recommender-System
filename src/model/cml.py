import torch
import torch.nn as nn
import torch.nn.functional as F
from .rs import normalize_embedding
from .model import init_param


# https://github.com/MogicianXD/CML_torch/blob/master/CML.py

class CML(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, features=None,clip_norm=1.0, margin=1.0, hidden_dim=128,
                 dropout=0.2, feature_l2_reg=0.0, feature_scale=1.0, use_bias=False):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.clip_norm = clip_norm
        self.margin = margin
        self.use_bias = use_bias

        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.item_embeddings = nn.Embedding(num_items, embed_dim)
        nn.init.normal_(self.user_embeddings.weight, std=1 / embed_dim ** 0.5)
        nn.init.normal_(self.item_embeddings.weight, std=1 / embed_dim ** 0.5)

        # Optional side features
        self.use_features = features is not None
        if self.use_features:
            self.register_buffer("features", torch.tensor(features, dtype=torch.float32))
            self.feature_proj = nn.Sequential(
                nn.Linear(features.shape[-1], hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim)
            )
            self.feature_scale = feature_scale
            self.feature_l2_reg = feature_l2_reg

        if self.use_bias:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))

        if self.use_features:
            self.item_embeddings.weight.data = self.feature_projection()

    def feature_projection(self):
        projected = self.feature_proj(self.features) * self.feature_scale
        return self.clip_by_norm(projected)

    def clip_by_norm(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        return x * (self.clip_norm / norm).clamp(max=1.0)

    def user_embedding(self, user_ids):
        return self.user_embeddings(user_ids)

    def item_embedding(self, item_ids):
        return self.item_embeddings(item_ids)

    def forward(self, user_ids, item_ids):
        u = self.user_embeddings(user_ids)
        v = self.item_embeddings(item_ids)
        dist = torch.sum((u - v) ** 2, dim=-1)  # squared Euclidean
        scores = -dist  # higher = more similar

        if self.use_bias:
            scores += self.user_bias(user_ids).squeeze(-1)
            scores += self.item_bias(item_ids).squeeze(-1)
            scores += self.global_bias

        return scores

    def full_score(self, user_ids):
        """Return scores of all items for each user."""
        u = self.user_embeddings(user_ids)             # [B, D]
        v_all = self.item_embeddings.weight            # [I, D]
        scores = -((u.unsqueeze(1) - v_all) ** 2).sum(dim=-1)  # [B, I]
        if self.use_bias:
            scores += self.item_bias.weight.T          # [1, I]
            scores += self.user_bias(user_ids)         # [B, 1]
            scores += self.global_bias
        return scores


def cml(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    model = CML(num_users, num_items, embedding_mode, **cfg['simplex'])
    model.apply(init_param)
    return model
