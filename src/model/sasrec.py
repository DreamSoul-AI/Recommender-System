import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .rs import normalize_embedding
from .layers import MLP


class SASRec(nn.Module):
    """SASRec: Self-Attentive Sequential Recommendation
    Args:
        num_users (int): Number of users.
        num_items (int): Number of items.
        embedding_mode (str): Normalization mode for embeddings.
        hidden_size (int): Dimension of embeddings and hidden layers.
        max_len (int): Maximum sequence length.
        num_blocks (int): Number of self-attention blocks.
        num_heads (int): Number of attention heads in MultiheadAttention.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, num_users, num_items, embedding_mode, hidden_size, max_len=50, num_blocks=2, num_heads=1, dropout_rate=0.5):
        super(SASRec, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_mode = embedding_mode
        self.hidden_size = hidden_size
        self.max_len = max_len

        # Embedding layers
        self.user_weight = nn.Embedding(num_users, hidden_size)
        self.item_weight = nn.Embedding(num_items + 1, hidden_size)  # +1 for padding index
        self.position_emb = nn.Embedding(max_len, hidden_size)

        self.emb_dropout = nn.Dropout(p=dropout_rate)

        # Transformer blocks
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_size, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_size, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rate))
            self.forward_layernorms.append(nn.LayerNorm(hidden_size, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_size, dropout_rate))

    def make_padding(self, seq, padding_idx):
        """Replace -100 (mask token) with padding_idx."""
        return torch.where(seq == -100, seq.new_full((1,), padding_idx), seq)

    def seq_forward(self, item_hist):
        """Processes item history through SASRec Transformer blocks."""
        item_hist = self.make_padding(item_hist, self.num_items)
        lengths = (item_hist != self.num_items).sum(dim=1).tolist()

        seq_embedding = self.item_weight(item_hist) * (self.hidden_size ** 0.5)

        # Add positional encoding
        batch_size, seq_len = item_hist.shape
        positions = torch.arange(seq_len, device=item_hist.device).repeat(batch_size, 1)
        seq_embedding += self.position_emb(positions)
        seq_embedding = self.emb_dropout(seq_embedding)

        # Mask for padding tokens
        timeline_mask = item_hist == self.num_items
        seq_embedding *= ~timeline_mask.unsqueeze(-1)

        # Self-attention mask
        attention_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=item_hist.device))

        for i in range(len(self.attention_layers)):
            seq_embedding = seq_embedding.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
            Q = self.attention_layernorms[i](seq_embedding)
            mha_outputs, _ = self.attention_layers[i](Q, seq_embedding, seq_embedding, attn_mask=attention_mask)
            seq_embedding = Q + mha_outputs
            seq_embedding = seq_embedding.transpose(0, 1)  # (batch_size, seq_len, hidden_size)

            seq_embedding = self.forward_layernorms[i](seq_embedding)
            seq_embedding = self.forward_layers[i](seq_embedding)
            seq_embedding *= ~timeline_mask.unsqueeze(-1)

        seq_output = self.last_layernorm(seq_embedding)
        return seq_output

    def user_embedding(self, user, item_hist):
        """Generates user embedding using historical interactions."""
        user_embedding = self.user_weight(user)
        item_hist_embedding = self.seq_forward(item_hist)
        item_hist_embedding = item_hist_embedding[:, -1, :]  # Last time step representation
        user_embedding = torch.cat([user_embedding, item_hist_embedding], dim=-1)
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


class PointWiseFeedForward(nn.Module):
    """Position-wise FeedForward layer used in Transformer blocks."""

    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        """Forward pass for feedforward network."""
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) + inputs
        return outputs


def sasrec(cfg):
    """Initializes SASRec model based on config."""
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    model = SASRec(num_users, num_items, embedding_mode, **cfg['youtubednn'])
    return model
