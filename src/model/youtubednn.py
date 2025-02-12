import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import init_param
from .rs import normalize_embedding

# from .basic.features import SparseFeature, SequenceFeature
# from .basic.layers import MLP, EmbeddingLayer

"""
https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/matching/youtube_dnn.py
"""


class MLP(nn.Module):
    """Multi Layer Perceptron Module, it is the most widely used module for
    learning feature. Note we default add `BatchNorm1d` and `Activation`
    `Dropout` for each `Linear` Module.

    Args:
        input dim (int): input size of the first Linear Layer.
        output_layer (bool): whether this MLP module is the output layer. If `True`, then append one Linear(*,1) module.
        dims (list): output size of Linear Layer (default=[]).
        dropout (float): probability of an element to be zeroed (default = 0.5).
        activation (str): the activation function, support `[sigmoid, relu, prelu, dice, softmax]` (default='relu').

    Shape:
        - Input: `(batch_size, input_dim)`
        - Output: `(batch_size, 1)` or `(batch_size, dims[-1])`
    """

    def __init__(self, input_dim, output_layer=True, dims=None, dropout=0):
        super().__init__()
        if dims is None:
            dims = []
        layers = list()
        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class YoutubeDNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_mode, hidden_size=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_mode = embedding_mode
        self.hidden_size = hidden_size
        self.user_weight = nn.Embedding(self.num_users, self.hidden_size)
        self.item_weight = nn.Embedding(self.num_items + 1, self.hidden_size)
        # self.user_bias = nn.Embedding(self.num_users, 1)
        # self.item_bias = nn.Embedding(self.num_items+1, 1)
        # self.pooling_layer = AveragePooling()
        # self.reset_parameters()
        # self.user_mlp = nn.Sequential(
        #     nn.Linear(self.hidden_size * 2, self.hidden_size),
        #     nn.ReLU()
        # )
        # TODO: user_params (dict): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)

    # def reset_parameters(self):
    #     nn.init.normal_(self.user_weight.weight, 0.0, 1e-4)
    #     nn.init.normal_(self.item_weight.weight, 0.0, 1e-4)
    #     # nn.init.constant_(self.user_bias.weight, 0.0)
    #     # nn.init.constant_(self.item_bias.weight, 0.0)
    #     return

    def user_embedding(self, user, item_hist): # TODO: need to fix this
        # Handle padding in item history
        # mask = item_hist == -100  # Create a mask for padding
        # item_hist = item_hist.clone()  # Avoid modifying the original tensor
        # item_hist[mask] = self.padding_idx  # Replace padding indices

        # Get embeddings for item history
        # item_hist_embedding = self.item_weight(item_hist)  # Shape: (batch_size, seq_len, embedding_dim)
        # pooling_item_hist_embedding = self.pooling_layer(item_hist_embedding,
        #                                                  mask.float().unsqueeze(1))  # Apply pooling with mask

        # Get user embeddings
        # initial_user_embedding = self.user_weight(user)  # Shape: (batch_size, embedding_dim)

        # Concatenate user and pooled item history embeddings
        # user_item_concat = torch.cat([initial_user_embedding, pooling_item_hist_embedding],
        #                              dim=-1)  # Shape: (batch_size, embedding_dim * 2)
        user_embedding = self.user_weight(user)
        user_embedding = self.user_mlp(user_embedding)  # [batch_size, 1, embed_dim]
        user_embedding = normalize_embedding(user_embedding, self.embedding_mode, 'user')
        # Pass through MLP and normalize
        # deep_user_embedding = self.user_mlp(user_item_concat).unsqueeze(1)  # Shape: (batch_size, 1, embedding_dim)
        # deep_user_embedding = F.normalize(deep_user_embedding, p=2, dim=2)  # L2 normalization along embedding dimension
        # i need to squeeze here 
        # return deep_user_embedding.squeeze(1)
        return user_embedding

    def item_embedding(self, item):
        item_embedding = self.item_weight(item)
        item_embedding = normalize_embedding(item_embedding, self.embedding_mode, 'item')
        return item_embedding

    def forward(self, user, item, item_hist):
        user_embedding = self.user_embedding(user, item_hist)
        user_embedding = self.dropout(user_embedding)
        item_embedding = self.item_embedding(item)
        if item_embedding.dim() == 2:
            item_embedding = item_embedding.unsqueeze(1)
        output_rating = torch.bmm(item_embedding, user_embedding.unsqueeze(-1)).squeeze(-1)
        return output_rating, user_embedding, item_embedding


# class AveragePooling(nn.Module):
#     """Pooling the sequence embedding matrix by `mean`.
#
#     Shape:
#         - Input
#             x: `(batch_size, seq_length, embed_dim)`
#             mask: `(batch_size, 1, seq_length)`
#         - Output: `(batch_size, embed_dim)`
#     """
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x, mask=None):
#         if mask == None:
#             return torch.mean(x, dim=1)
#         else:
#             sum_pooling_matrix = torch.bmm(mask, x).squeeze(1)
#             non_padding_length = mask.sum(dim=-1)
#             return sum_pooling_matrix / (non_padding_length.float() + 1e-16)


def youtubednn(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    model = YoutubeDNN(num_users, num_items, embedding_mode, **cfg['youtubednn'])
    model.apply(init_param)
    return model
