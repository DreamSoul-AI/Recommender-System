import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import init_param
from .rs import normalize_embedding
from .layers import MLP, AveragePooling


class FaceBookDSSM(nn.Module):
    """Modified FaceBookDSSM to align with YoutubeDNN structure.

    Args:
        num_users (int): Number of users.
        num_items (int): Number of items.
        embedding_mode (str): Normalization mode for embeddings.
        hidden_size (int or list): Dimension of embeddings.
        temperature (float): Scaling factor for cosine similarity.
    """

    def __init__(self, num_users, num_items, embedding_mode, hidden_size, temperature=1.0):
        super(FaceBookDSSM, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_mode = embedding_mode
        self.hidden_size = hidden_size[-1] if isinstance(hidden_size, list) else hidden_size
        self.temperature = temperature

        # Embedding layers
        self.user_weight = nn.Embedding(num_users, self.hidden_size)
        self.item_weight = nn.Embedding(num_items + 1, self.hidden_size)  # +1 for padding index
        self.pooling_layer = AveragePooling()

        # MLP layers
        self.user_mlp = MLP(self.hidden_size * 2, output_layer=False, dims=[self.hidden_size, self.hidden_size])

    def make_padding(self, hist, padding_idx):
        """Replace -100 (mask token) with padding_idx."""
        return torch.where(hist == -100, hist.new_full((1,), padding_idx), hist)

    def user_embedding(self, user, item_hist):
        """Generates user embedding using historical interactions."""
        mask = item_hist == -100
        item_hist = self.make_padding(item_hist, self.num_items)

        # AveragePooling
        item_hist_embedding = self.item_weight(item_hist)
        item_hist_embedding = self.pooling_layer(item_hist_embedding, mask)

      
        user_embedding = self.user_weight(user)


        user_embedding = torch.cat([user_embedding, item_hist_embedding], dim=-1)
        user_embedding = self.user_mlp(user_embedding)
        user_embedding = normalize_embedding(user_embedding, self.embedding_mode, 'user')
        return user_embedding

    def item_embedding(self, item):
        """Generates item embeddings."""
        item_embedding = self.item_weight(item)
        item_embedding = normalize_embedding(item_embedding, self.embedding_mode, 'item')
        return item_embedding

    def forward(self, user, item, item_hist):
        """Compute interaction scores for user and item."""
        user_embedding = self.user_embedding(user, item_hist)
        item_embedding = self.item_embedding(item)

        if item_embedding.dim() == 2:
            item_embedding = item_embedding.unsqueeze(1)  # (batch_size, 1, hidden_size)

        output_rating = torch.bmm(item_embedding, user_embedding.unsqueeze(-1)).squeeze(-1)
        return output_rating, user_embedding, item_embedding


def dssm_facebook(cfg):
    """Initializes FaceBookDSSM model based on config."""
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']

    model = FaceBookDSSM(num_users, num_items, embedding_mode, **cfg['youtubednn'])
    model.apply(init_param)
    return model
