import torch
import torch.nn as nn
import torch.nn.functional as F

# from .basic.features import SparseFeature, SequenceFeature
# from .basic.layers import MLP, EmbeddingLayer

"""
https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/matching/mind.py
"""

    
    
class MIND(nn.Module):
    def __init__(self, num_users, num_items, hidden_size, interest_num=4):
        super().__init__()
        self.num_users = num_users
        
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.padding_idx = num_items 
        self.interest_num = interest_num
        
        self.user_weight = nn.Embedding(self.num_users, self.hidden_size)
        self.item_weight = nn.Embedding(self.num_items+1, self.hidden_size)
        self.max_length = 160
        self.capsule = CapsuleNetwork(self.hidden_size,self.max_length,bilinear_type=0,interest_num=self.interest_num)

        self.convert_user_weight = nn.Parameter(torch.rand(self.hidden_size*2, self.hidden_size), requires_grad=True)
        self.reset_parameters()

        
    def reset_parameters(self):
        nn.init.normal_(self.user_weight.weight, 0.0, 1e-4)
        nn.init.normal_(self.item_weight.weight, 0.0, 1e-4)

        return
    
    def user_embedding(self, user, item_hist):
        input_user =self.user_weight(user).unsqueeze(1)
        input_user = input_user.expand([input_user.shape[0], self.interest_num, input_user.shape[-1]]) #[256, 4, 16]
        # Handle padding in item history
       
        mask = item_hist == -100  # Create a mask for padding
        item_hist = item_hist.clone()  # Avoid modifying the original tensor
        item_hist[mask] = self.padding_idx  # Replace padding indices 
        item_hist_embedding = self.item_weight(item_hist)
        
        multi_interest_emb = self.capsule(item_hist_embedding,mask) #[256, 4, 16]
        input_user = torch.cat([input_user,multi_interest_emb],dim=-1) #[256, 4, 32]
        user_embedding = torch.matmul(input_user,self.convert_user_weight)
        user_embedding = F.normalize(user_embedding, p=2, dim=-1) 
    
        return user_embedding #256*4*16
    
    def item_embedding(self, item):

        return self.item_weight(item)
    
    def forward(self, user, item, item_hist):
        
        user_embedding = self.user_embedding(user, item_hist)
        item_embedding = self.item_embedding(item)
        
        pos_item_embedding = item_embedding[:,0,:]
        dot_res = torch.bmm(user_embedding, pos_item_embedding.squeeze(1).unsqueeze(-1))
        k_index = torch.argmax(dot_res, dim=1)
        best_interest_emb = torch.rand(user_embedding.shape[0], user_embedding.shape[2]).to(user_embedding.device)
        for k in range(user_embedding.shape[0]):
            best_interest_emb[k, :] = user_embedding[k, k_index[k], :]
        best_interest_emb = best_interest_emb.unsqueeze(1)

        output_rating = torch.mul(best_interest_emb, item_embedding).sum(dim=-1)

        return output_rating, user_embedding, item_embedding  
        
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
            self.w = nn.Parameter(torch.Tensor(1, self.seq_len, self.interest_num * self.embedding_dim, self.embedding_dim))

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

        for i in range(self.routing_times):  # 动态路由传播3次
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

def mind(cfg):
    num_users = cfg['stats']['num_users']
    num_items = cfg['stats']['num_items']
    hidden_size = 16 
    model = MIND(num_users, num_items, hidden_size)
    return model