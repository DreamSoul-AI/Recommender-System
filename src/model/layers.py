import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CapsuleNetwork(nn.Module):
    """CapsuleNetwork mentioned in the Comirec and MIND paper.

    Args:
        hidden_size (int): embedding dim of item embedding
        seq_len (int): length of the item sequence
        bilinear_type (int): 0 for MIND, 2 for ComirecDR
        interest_num (int): num of interest
        routing_times (int): routing times

    Shape:
        - Input: seq_emb : (batch,seq,emb)
                 mask : (batch,seq,1)
        - Output: `(batch_size, interest_num, embedding_dim)`

    """

    def __init__(self, embedding_dim, seq_len, bilinear_type=2, interest_num=4, routing_times=3, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.embedding_dim = embedding_dim  # h
        self.seq_len = seq_len  # s
        self.bilinear_type = bilinear_type
        self.interest_num = interest_num
        self.routing_times = routing_times

        self.relu_layer = relu_layer
        self.stop_grad = True
        self.relu = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), nn.ReLU())
        if self.bilinear_type == 0:  # MIND
            self.linear = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        elif self.bilinear_type == 1:
            self.linear = nn.Linear(self.embedding_dim, self.embedding_dim * self.interest_num, bias=False)
        else:
            self.w = nn.Parameter(
                torch.Tensor(1, self.seq_len, self.interest_num * self.embedding_dim, self.embedding_dim))

    def forward(self, item_eb, mask):
        if self.bilinear_type == 0:
            item_eb_hat = self.linear(item_eb)
            item_eb_hat = item_eb_hat.repeat(1, 1, self.interest_num)
        elif self.bilinear_type == 1:
            item_eb_hat = self.linear(item_eb)
        else:
            u = torch.unsqueeze(item_eb, dim=2)
            item_eb_hat = torch.sum(self.w[:, :self.seq_len, :, :] * u, dim=3)

        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.seq_len, self.interest_num, self.embedding_dim))
        item_eb_hat = torch.transpose(item_eb_hat, 1, 2).contiguous()
        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.interest_num, self.seq_len, self.embedding_dim))

        if self.stop_grad:
            item_eb_hat_iter = item_eb_hat.detach()
        else:
            item_eb_hat_iter = item_eb_hat

        if self.bilinear_type > 0:
            capsule_weight = torch.zeros(item_eb_hat.shape[0],
                                         self.interest_num,
                                         self.seq_len,
                                         device=item_eb.device,
                                         requires_grad=False)
        else:
            capsule_weight = torch.randn(item_eb_hat.shape[0],
                                         self.interest_num,
                                         self.seq_len,
                                         device=item_eb.device,
                                         requires_grad=False)

        interest_capsule = None
        for i in range(self.routing_times):
            atten_mask = torch.unsqueeze(mask, 1).repeat(1, self.interest_num, 1)
            paddings = torch.zeros_like(atten_mask, dtype=torch.float)

            capsule_softmax_weight = F.softmax(capsule_weight, dim=-1)
            capsule_softmax_weight = torch.where(torch.eq(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = torch.unsqueeze(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat_iter)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = torch.matmul(item_eb_hat_iter, torch.transpose(interest_capsule, 2, 3).contiguous())
                delta_weight = torch.reshape(delta_weight, (-1, self.interest_num, self.seq_len))
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = torch.reshape(interest_capsule, (-1, self.interest_num, self.embedding_dim))

        if self.relu_layer:
            interest_capsule = self.relu(interest_capsule)

        return interest_capsule


class MultiInterestSA(nn.Module):
    """MultiInterest Attention mentioned in the Comirec paper.

    Args:
        embedding_dim (int): embedding dim of item embedding
        interest_num (int): num of interest
        hidden_dim (int): hidden dim

    Shape:
        - Input: seq_emb : (batch,seq,emb)
                 mask : (batch,seq,1)
        - Output: `(batch_size, interest_num, embedding_dim)`

    """

    def __init__(self, embedding_dim, interest_num, hidden_dim=None):
        super(MultiInterestSA, self).__init__()
        self.embedding_dim = embedding_dim
        self.interest_num = interest_num
        if hidden_dim == None:
            self.hidden_dim = self.embedding_dim * 4
        self.W1 = torch.nn.Parameter(torch.rand(self.embedding_dim, self.hidden_dim), requires_grad=True)
        self.W2 = torch.nn.Parameter(torch.rand(self.hidden_dim, self.interest_num), requires_grad=True)
        self.W3 = torch.nn.Parameter(torch.rand(self.embedding_dim, self.embedding_dim), requires_grad=True)

    def forward(self, seq_emb, mask=None):
        H = torch.einsum('bse, ed -> bsd', seq_emb, self.W1).tanh()
        if mask != None:
            A = torch.einsum('bsd, dk -> bsk', H, self.W2) + -1.e9 * (1 - mask.float())
            A = F.softmax(A, dim=1)
        else:
            A = F.softmax(torch.einsum('bsd, dk -> bsk', H, self.W2), dim=1)
        A = A.permute(0, 2, 1)
        multi_interest_emb = torch.matmul(A, seq_emb)
        return multi_interest_emb
