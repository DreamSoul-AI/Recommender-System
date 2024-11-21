import torch
import torch.nn as nn
import torch.nn.functional as F
# from .basic.features import SparseFeature, SequenceFeature
# from .basic.layers import MLP, EmbeddingLayer
import pdb

"""
https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/matching/youtube_dnn.py
"""

class YoutubeDNN(nn.Module):
    def __init__(self, num_users, num_items, hidden_size):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.padding_idx = num_items 
        
        self.user_weight = nn.Embedding(self.num_users, self.hidden_size)
        self.item_weight = nn.Embedding(self.num_items+1, self.hidden_size)
        # self.user_bias = nn.Embedding(self.num_users, 1)
        # self.item_bias = nn.Embedding(self.num_items+1, 1)
        self.pooling_layer = AveragePooling()
        self.reset_parameters()

        self.user_mlp = nn.Sequential(
                nn.Linear(self.hidden_size*2, self.hidden_size),
                nn.ReLU()  
            )
        
    def reset_parameters(self):
        nn.init.normal_(self.user_weight.weight, 0.0, 1e-4)
        nn.init.normal_(self.item_weight.weight, 0.0, 1e-4)
        # nn.init.constant_(self.user_bias.weight, 0.0)
        # nn.init.constant_(self.item_bias.weight, 0.0)
        return
    
    def user_tower(self, user, item_hist):
        # Handle padding in item history
        mask = item_hist == -100  # Create a mask for padding
        item_hist = item_hist.clone()  # Avoid modifying the original tensor
        item_hist[mask] = self.padding_idx  # Replace padding indices
    
        # Get embeddings for item history
        item_hist_embedding = self.item_weight(item_hist)  # Shape: (batch_size, seq_len, embedding_dim)
        pooling_item_hist_embedding = self.pooling_layer(item_hist_embedding, mask.float().unsqueeze(1))  # Apply pooling with mask

        # Get user embeddings
        initial_user_embedding = self.user_weight(user)  # Shape: (batch_size, embedding_dim)

        # Concatenate user and pooled item history embeddings
        user_item_concat = torch.cat([initial_user_embedding, pooling_item_hist_embedding], dim=-1)  # Shape: (batch_size, embedding_dim * 2)

        # Pass through MLP and normalize
        deep_user_embedding = self.user_mlp(user_item_concat).unsqueeze(1)  # Shape: (batch_size, 1, embedding_dim)
        deep_user_embedding = F.normalize(deep_user_embedding, p=2, dim=2)  # L2 normalization along embedding dimension

        return deep_user_embedding
    
    def item_tower(self, item):

        return self.item_weight(item)
    
    def forward(self, user, item, rating, item_hist):

        # Get embeddings from user and item towers
        user_embedding = self.user_tower(user, item_hist)
        item_embedding = self.item_tower(item)
        #y = torch.mul(user_embedding, item_embedding).sum(dim=2)
        return user_embedding.squeeze(1), item_embedding

class AveragePooling(nn.Module):
    """Pooling the sequence embedding matrix by `mean`.
    
    Shape:
        - Input
            x: `(batch_size, seq_length, embed_dim)`
            mask: `(batch_size, 1, seq_length)`
        - Output: `(batch_size, embed_dim)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask == None:
            return torch.mean(x, dim=1)
        else:
            sum_pooling_matrix = torch.bmm(mask, x).squeeze(1)
            non_padding_length = mask.sum(dim=-1)
            return sum_pooling_matrix / (non_padding_length.float() + 1e-16)                     

def youtubednn(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    hidden_size = cfg['youtubednn']['hidden_size']
    model = YoutubeDNN(num_users, num_items, hidden_size)
    return model

# input_user = self.embedding({'user':user}, user_features, squeeze_dim=True)
