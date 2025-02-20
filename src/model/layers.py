import torch
import torch.nn as nn

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
        if mask is None:
            x = torch.mean(x, dim=1)
        else:
            mask = mask.float().unsqueeze(1)
            sum_pooling_matrix = torch.bmm(mask, x).squeeze(1)
            non_padding_length = mask.sum(dim=-1)
            x = sum_pooling_matrix / (non_padding_length.float() + 1e-16)
        return x
