import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import init_param
from .rs import normalize_embedding
from .layers import MLP, AveragePooling


# https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/matching/sasrec.py

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SASRec(nn.Module):
    def __init__(self, num_users, num_items, embedding_mode, hidden_size, max_length=50, dropout_rate=0., num_blocks=2,
                 num_heads=1):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_mode = embedding_mode
        self.hidden_size = hidden_size
        self.max_len = max_length
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        # Embedding layers
        self.user_weight = nn.Embedding(num_users, self.hidden_size)
        self.item_weight = nn.Embedding(num_items + 1, self.hidden_size)  # +1 for padding index
        self.position_emb = nn.Embedding(max_length['user'] + 1, self.hidden_size)
        # self.pooling_layer = AveragePooling()

        # MLP layers
        # self.user_mlp = MLP(self.hidden_size * 2, output_layer=False, dims=[self.hidden_size, self.hidden_size])
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = torch.nn.MultiheadAttention(self.hidden_size, num_heads, dropout_rate, batch_first=True)
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(self.hidden_size, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def make_padding(self, hist, padding_idx):
        """Replace -100 (mask token) with padding_idx."""
        return torch.where(hist == -100, hist.new_full((1,), padding_idx), hist)

    def user_embedding(self, user, item_hist):
        """Generates user embedding using historical interactions."""
        mask = item_hist == -100
        item_hist = self.make_padding(item_hist, self.num_items)
        item_hist_embedding = self.item_weight(item_hist)
        # item_hist_embedding = self.pooling_layer(item_hist_embedding, mask)

        item_hist_embedding *= self.hidden_size ** 0.5
        # item_hist_embedding = item_hist_embedding.squeeze()  # (batch_size, max_len, embed_dim)
        # positions = np.tile(np.array(range(x.shape[1])), [x.shape[0], 1])
        # embed_x_feature += self.position_emb(torch.LongTensor(positions))
        pos_embedding = self.position_emb(torch.arange(item_hist_embedding.size(1), device=item_hist.device))
        item_hist_embedding += pos_embedding
        item_hist_embedding = self.emb_dropout(item_hist_embedding)

        # timeline_mask = torch.BoolTensor(x == 0)
        # embed_x_feature *= ~timeline_mask.unsqueeze(-1)
        attention_mask = torch.tril(torch.ones((item_hist_embedding.size(1), item_hist_embedding.size(1)),
                                                dtype=torch.bool, device=item_hist.device), diagonal=-1)
        for i in range(len(self.attention_layers)):
            # item_hist_embedding = torch.transpose(item_hist_embedding, 0, 1)
            Q = self.attention_layernorms[i](item_hist_embedding)
            mha_outputs, _ = self.attention_layers[i](Q, item_hist_embedding, item_hist_embedding,
                                                      attn_mask=attention_mask)
            item_hist_embedding = Q + mha_outputs
            # embed_x_feature = torch.transpose(embed_x_feature, 0, 1)
            item_hist_embedding = self.forward_layernorms[i](item_hist_embedding)
            item_hist_embedding = self.forward_layers[i](item_hist_embedding)
            print(item_hist_embedding.size())
            exit()
            embed_x_feature *= ~timeline_mask.unsqueeze(-1)

        user_embedding = self.last_layernorm(embed_x_feature)

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


def sasrec(cfg):
    """Initializes SASRec model based on config."""
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    max_length = cfg['max_length']
    model = SASRec(num_users, num_items, embedding_mode, max_length=max_length, **cfg['sasrec'])
    model.apply(init_param)
    return model
