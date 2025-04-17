import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import init_param
from .rs import normalize_embedding


class ComiRecDR(nn.Module):
    def __init__(self, num_users, num_items, embedding_mode, max_length, hidden_size, interest_num):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_mode = embedding_mode
        self.max_length = max_length['user']
        self.interest_num = interest_num
        self.hidden_size = hidden_size

        # User and item embeddings
        self.user_embedding = nn.Embedding(self.num_users, self.hidden_size)
        self.item_embedding = nn.Embedding(self.num_items + 1, self.hidden_size)

        # Capsule Network
        self.capsule = CapsuleNetwork(
            embedding_dim=self.hidden_size,
            seq_len=self.max_length,
            bilinear_type=2,
            interest_num=self.interest_num
        )

        # User tower projection layer
        self.convert_user_weight = nn.Linear(
            self.hidden_size * 2, self.hidden_size, bias=False
        )

        self.mode = None

    def forward(self, x):
        user_embedding = self.user_tower(x)
        item_embedding = self.item_tower(x)

        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding

        pos_item_embedding = item_embedding[:, 0, :]
        dot_res = torch.bmm(user_embedding, pos_item_embedding.unsqueeze(-1))
        k_index = torch.argmax(dot_res, dim=1)

        best_interest_emb = torch.stack(
            [user_embedding[i, k_index[i], :] for i in range(user_embedding.size(0))], dim=0
        ).unsqueeze(1)

        y = torch.mul(best_interest_emb, item_embedding).sum(dim=-1)
        return y

    def user_tower(self, x):
        if self.mode == "item":
            return None

        user_ids = x['user_id']
        history_items = x['hist_item_id']

        user_emb = self.user_embedding(user_ids)
        user_emb = user_emb.unsqueeze(1).expand(-1, self.interest_num, -1)

        history_emb = self.item_embedding(history_items)
        mask = (history_items > 0).long()

        multi_interest_emb = self.capsule(history_emb, mask)

        concat_user_emb = torch.cat([user_emb, multi_interest_emb], dim=-1)
        user_embedding = self.convert_user_weight(concat_user_emb)
        user_embedding = F.normalize(user_embedding, p=2, dim=-1)
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None

        pos_items = x['item_id']
        pos_emb = self.item_embedding(pos_items).unsqueeze(1)
        pos_emb = F.normalize(pos_emb, p=2, dim=-1)

        if self.mode == "item":
            return pos_emb.squeeze(1)

        neg_items = x['neg_item_id']
        neg_emb = self.item_embedding(neg_items).unsqueeze(1)
        neg_emb = F.normalize(neg_emb, p=2, dim=-1)

        return torch.cat((pos_emb, neg_emb), dim=1)


class CapsuleNetwork(nn.Module):
    def __init__(self, embedding_dim, seq_len, bilinear_type=2, interest_num=4, routing_times=3, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.interest_num = interest_num
        self.routing_times = routing_times
        self.relu_layer = relu_layer
        self.stop_grad = True
        self.relu = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), nn.ReLU())

        if self.bilinear_type == 0:
            self.linear = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        elif self.bilinear_type == 1:
            self.linear = nn.Linear(self.embedding_dim, self.embedding_dim * self.interest_num, bias=False)
        else:
            self.w = nn.Parameter(
                torch.Tensor(1, self.seq_len, self.interest_num * self.embedding_dim, self.embedding_dim))
            nn.init.xavier_uniform_(self.w)

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

        item_eb_hat_iter = item_eb_hat.detach() if self.stop_grad else item_eb_hat

        capsule_weight = torch.zeros(item_eb_hat.shape[0], self.interest_num, self.seq_len,
                                     device=item_eb.device, requires_grad=False)

        interest_capsule = None
        for i in range(self.routing_times):
            atten_mask = torch.unsqueeze(mask, 1).repeat(1, self.interest_num, 1)
            paddings = torch.zeros_like(atten_mask, dtype=torch.float)

            capsule_softmax_weight = F.softmax(capsule_weight, dim=-1)
            capsule_softmax_weight = torch.where(torch.eq(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = torch.unsqueeze(capsule_softmax_weight, 2)

            if i < self.routing_times - 1:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat_iter)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, keepdim=True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = torch.matmul(item_eb_hat_iter, torch.transpose(interest_capsule, 2, 3).contiguous())
                delta_weight = torch.reshape(delta_weight, (-1, self.interest_num, self.seq_len))
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, keepdim=True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = torch.reshape(interest_capsule, (-1, self.interest_num, self.embedding_dim))
        if self.relu_layer:
            interest_capsule = self.relu(interest_capsule)
        return interest_capsule


def comirec(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    embedding_mode = cfg['embedding_mode']
    max_length = cfg['max_length']
    model = ComiRecDR(num_users, num_items, embedding_mode, max_length, **cfg['comirec'])
    model.apply(init_param)
    return model
