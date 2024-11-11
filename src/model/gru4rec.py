import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torch
import torch.nn as nn
import pdb

"""
rewrite from https://github.com/reczoo/RecBox/blob/main/recbox/third_party/rechub/models/matching/gru4rec.py
"""

class GRU4Rec(nn.Module):
    def __init__(self, num_users, num_items, hidden_size, num_layers=2):
        super().__init__()
        self.num_users = num_users
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Increase num_items by 1 to include a padding index
        self.num_items = num_items + 1  # Add 1 for the padding index
        self.padding_idx = num_items    # Padding index is set to the last index

        # Embedding layers for users and items
        self.user_embedding_layer = nn.Embedding(num_users, hidden_size)
        self.item_embedding_layer = nn.Embedding(
            self.num_items, hidden_size, padding_idx=self.padding_idx
        )
        
        self.user_mlp = nn.Sequential(
                nn.Linear(self.hidden_size*2, self.hidden_size),
                nn.ReLU()  
            )

        # GRU layer for modeling user behavior sequences
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize embeddings
        nn.init.normal_(self.user_embedding_layer.weight, 0.0, 1e-4)
        nn.init.normal_(self.item_embedding_layer.weight, 0.0, 1e-4)
        # Ensure the padding index in item embeddings is zero
        with torch.no_grad():
            self.item_embedding_layer.weight[self.padding_idx].fill_(0)

    def user_embedding(self, user_ids, item_history):
        
        item_history = item_history.clone()
        item_history[item_history == -100] = self.padding_idx
        user_emb = self.user_embedding_layer(user_ids)  # Shape: [batch_size, hidden_size]

        item_hist_emb = self.item_embedding_layer(item_history)  # Shape: [batch_size, seq_len, hidden_size]

        lengths = (item_history != self.padding_idx).sum(dim=1)  # Shape: [batch_size]

        packed_emb = nn.utils.rnn.pack_padded_sequence(
            item_hist_emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.gru(packed_emb)

        # hidden shape: [num_layers, batch_size, hidden_size]
        # Use the last layer's hidden state
        hidden = hidden[-1]  # Shape: [batch_size, hidden_size]
        user_embedding = torch.cat([user_emb, hidden],dim=-1)
        user_embedding = self.user_mlp(user_embedding)
        user_embedding = F.normalize(user_embedding, p=2, dim=-1)  # L2 normalize
        
        return user_embedding

    def item_embedding(self, item_ids):
        item_embedding = self.item_embedding_layer(item_ids)  # Shape: [batch_size, hidden_size]
        item_embedding = F.normalize(item_embedding, p=2, dim=-1)
        return item_embedding

    def forward(self, user_ids, item_ids, ratings=None, item_history=None):

        user_emb = self.user_embedding(user_ids, item_history)
        item_emb = self.item_embedding(item_ids)
        pdb.set_trace()
        return user_emb, item_emb

def gru4rec(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    hidden_size = cfg['mf']['hidden_size']
    num_layers = 2
    model = GRU4Rec(num_users, num_items, hidden_size, num_layers)
    return model
