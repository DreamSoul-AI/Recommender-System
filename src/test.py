import torch
import numpy as np
import scipy.sparse as sp

from recbole.model.general_recommender.ngcf import NGCF
from recbole.data.interaction import Interaction
from recbole.config import Config


# Monkey patch get_norm_adj_mat to avoid .update() on dok_matrix
def patched_get_norm_adj_mat(self):
    A = sp.dok_matrix(
        (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
    )
    inter_M = self.interaction_matrix
    inter_M_t = self.interaction_matrix.transpose()

    for u, i in zip(inter_M.row, inter_M.col):
        A[u, i + self.n_users] = 1.0
    for i, u in zip(inter_M_t.row, inter_M_t.col):
        A[i + self.n_users, u] = 1.0

    sumArr = (A > 0).sum(axis=1)
    diag = np.array(sumArr.flatten())[0] + 1e-7
    diag = np.power(diag, -0.5)
    D = sp.diags(diag)
    L = D @ A @ D
    L = sp.coo_matrix(L)
    i = torch.LongTensor(np.array([L.row, L.col]))
    data = torch.FloatTensor(L.data)
    return torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

NGCF.get_norm_adj_mat = patched_get_norm_adj_mat


# Create a dummy config and dataset
def create_dummy_config_and_dataset(n_users=4, n_items=5):
    config_dict = {
        'model': 'NGCF',
        'dataset': 'dummy_dataset',
        'embedding_size': 8,
        'hidden_size_list': [8, 8],
        'node_dropout': 0.1,
        'message_dropout': 0.1,
        'reg_weight': 1e-4,
        'epochs': 1,
        'device': 'cpu',
    }

    config = Config(model='NGCF', config_dict=config_dict)

    inter_matrix = sp.coo_matrix(
        (
            np.ones(6),
            (
                np.array([0, 1, 2, 1, 3, 0]),  # users
                np.array([0, 1, 2, 3, 4, 2]),  # items
            ),
        ),
        shape=(n_users, n_items),
    )

    class DummyDataset:
        def __init__(self):
            self.n_users = n_users
            self.n_items = n_items
            self.field2id_token = {
                'user_id': [str(i) for i in range(n_users)],
                'item_id': [str(i) for i in range(n_items)],
            }

        def num(self, field):
            return self.n_users if field == 'user_id' else self.n_items

        def inter_matrix(self, form="coo"):
            return inter_matrix

    return config, DummyDataset()


def run_test():
    config, dataset = create_dummy_config_and_dataset()
    model = NGCF(config, dataset)

    print("=== Model initialized ===")
    print(f"user_embedding shape: {model.user_embedding.weight.shape}")
    print(f"item_embedding shape: {model.item_embedding.weight.shape}")

    print("\n=== norm_adj_matrix ===")
    norm_adj = model.norm_adj_matrix
    print(f"norm_adj_matrix shape: {norm_adj.shape}")
    print(f"nnz: {norm_adj._nnz()}")

    print("\n=== eye_matrix ===")
    print(f"eye_matrix shape: {model.eye_matrix.shape}")
    print(f"nnz: {model.eye_matrix._nnz()}")

    ego_embeddings = model.get_ego_embeddings()
    print("\n=== ego_embeddings ===")
    print(f"ego_embeddings shape: {ego_embeddings.shape}")

    print("\n=== Forward Pass ===")
    user_emb, item_emb = model.forward()
    print(f"user_all_embeddings shape: {user_emb.shape}")
    print(f"item_all_embeddings shape: {item_emb.shape}")

    print("\n=== calculate_loss ===")
    interaction_dict = {
        'user_id': torch.LongTensor([0, 1]),
        'item_id': torch.LongTensor([1, 2]),
        'neg_item_id': torch.LongTensor([3, 4]),
    }
    interaction = Interaction(interaction_dict)
    loss = model.calculate_loss(interaction)
    print(f"Loss: {loss.item()}")

    print("\n=== predict ===")
    scores = model.predict(interaction)
    print(f"Scores: {scores}")

    print("\n=== full_sort_predict ===")
    full_sort_scores = model.full_sort_predict(interaction)
    print(f"Full sort scores shape: {full_sort_scores.shape}")
    print(f"Scores: {full_sort_scores}")


if __name__ == "__main__":
    run_test()
