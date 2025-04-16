import torch
import torch.nn as nn

"""
https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/matching/youtube_dnn.py
"""


class MLP(nn.Module):
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


class AveragePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask is None:
            x = torch.mean(x, dim=1)
        else:
            mask = mask.float().unsqueeze(1)
            sum_pooling_matrix = torch.bmm(mask, x).squeeze(1)
            non_padding_length = mask.sum(dim=-1)
            x = sum_pooling_matrix / (non_padding_length.float() + 1e-16)
        return x


class SENETLayer(nn.Module):
    def __init__(self, num_fields, reduction_ratio=4):
        super(SENETLayer, self).__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.mlp = nn.Sequential(nn.Linear(num_fields, reduced_size, bias=False),
                                 nn.ReLU(),
                                 nn.Linear(reduced_size, num_fields, bias=False),
                                 nn.ReLU())

    def forward(self, x):
        z = torch.mean(x, dim=1, out=None)
        a = self.mlp(z)
        v = x * a.unsqueeze(1)
        return v
