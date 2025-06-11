import torch
import torch.nn as nn
import torch.nn.functional as F
from .rs import normalize_embedding
from .model import init_param


# https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/general_recommender/neumf.py
# https://github.com/AmazingDD/daisyRec/blob/dev/daisy/model/NeuMFRecommender.py

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=32, mlp_dim=64, mlp_layers=[128, 64], dropout=0.0,
                 mf_train=True, mlp_train=True, use_bias=False):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.mf_train = mf_train
        self.mlp_train = mlp_train
        self.use_bias = use_bias

        # Embeddings
        if mf_train:
            self.user_mf = nn.Embedding(num_users, mf_dim)
            self.item_mf = nn.Embedding(num_items, mf_dim)

        if mlp_train:
            self.user_mlp = nn.Embedding(num_users, mlp_dim)
            self.item_mlp = nn.Embedding(num_items, mlp_dim)

            mlp_input_dim = 2 * mlp_dim
            mlp_layers_ = []
            for dim in mlp_layers:
                mlp_layers_.append(nn.Linear(mlp_input_dim, dim))
                mlp_layers_.append(nn.ReLU())
                mlp_layers_.append(nn.Dropout(p=dropout))
                mlp_input_dim = dim
            self.mlp_layers = nn.Sequential(*mlp_layers_)

        if mf_train and mlp_train:
            fusion_dim = mf_dim + mlp_layers[-1]
        elif mf_train:
            fusion_dim = mf_dim
        elif mlp_train:
            fusion_dim = mlp_layers[-1]

        self.predict_layer = nn.Linear(fusion_dim, 1)

        if use_bias:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))

    def user_embedding(self, user_ids):
        """Returns final user embedding after GMF + MLP fusion."""
        emb_list = []
        if self.mf_train:
            emb_list.append(self.user_mf(user_ids))
        if self.mlp_train:
            emb_list.append(self.user_mlp(user_ids))
        return tuple(emb_list) if len(emb_list) > 1 else emb_list[0]

    def item_embedding(self, item_ids):
        """Returns final item embedding after GMF + MLP fusion."""
        emb_list = []
        if self.mf_train:
            emb_list.append(self.item_mf(item_ids))
        if self.mlp_train:
            emb_list.append(self.item_mlp(item_ids))
        return tuple(emb_list) if len(emb_list) > 1 else emb_list[0]

    def forward(self, user_ids, item_ids):
        """Computes prediction scores using fused representations."""
        emb_mf_user = self.user_mf(user_ids) if self.mf_train else None
        emb_mf_item = self.item_mf(item_ids) if self.mf_train else None
        emb_mlp_user = self.user_mlp(user_ids) if self.mlp_train else None
        emb_mlp_item = self.item_mlp(item_ids) if self.mlp_train else None

        if self.mf_train:
            mf_output = emb_mf_user * emb_mf_item  # element-wise product
        if self.mlp_train:
            mlp_input = torch.cat([emb_mlp_user, emb_mlp_item], dim=-1)
            mlp_output = self.mlp_layers(mlp_input)

        if self.mf_train and self.mlp_train:
            output = torch.cat([mf_output, mlp_output], dim=-1)
        elif self.mf_train:
            output = mf_output
        else:
            output = mlp_output

        pred = self.predict_layer(output).squeeze(-1)

        if self.use_bias:
            pred += self.user_bias(user_ids).squeeze(-1)
            pred += self.item_bias(item_ids).squeeze(-1)
            pred += self.global_bias

        return pred


def neumf(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    model = NeuMF(num_users, num_items, embedding_mode, **cfg['simplex'])
    model.apply(init_param)
    return model
