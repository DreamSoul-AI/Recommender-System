import torch
import torch.nn.functional as F
from collections import defaultdict
from config import cfg
from module import recur


def make_metric(split):
    metric_name = {k: [] for k in split}
    if cfg['data_name'] in ['ML100K']:
        if cfg['target_mode'] == 'explicit':
            best = float('inf')
            best_direction = 'down'
            best_metric_name = 'MSE'
            for k in metric_name:
                metric_name[k].extend(['Loss', 'MSE'])
        elif cfg['target_mode'] == 'implicit':
            best = -float('inf')
            best_direction = 'up'
            best_metric_name = 'NDCG'
            for k in metric_name:
                metric_name[k].extend(['Loss', 'NDCG'])
        else:
            raise ValueError('Not valid target mode')
    else:
        raise ValueError('Not valid data name')
    metric = Metric(metric_name, best, best_direction, best_metric_name)
    return metric


def Accuracy(output, target, topk=1):
    with torch.no_grad():
        if target.dtype != torch.int64:
            target = (target.topk(1, -1, True, True)[1]).view(-1)
        batch_size = torch.numel(target)
        pred_k = output.topk(topk, -1, True, True)[1]
        correct_k = pred_k.eq(target.unsqueeze(-1).expand_as(pred_k)).float().sum()
        acc = (correct_k * (100.0 / batch_size)).item()
    return acc


def MSE(output, target):
    with torch.no_grad():
        mse = F.mse_loss(output, target).item()
    return mse


# def _setup_matrix(output, target, user, item):
#     user, user_idx = torch.unique(user, return_inverse=True)
#     item, item_idx = torch.unique(item, return_inverse=True)
#     num_users, num_items = len(user), len(item)
#
#     if cfg['data_mode'] == 'user':
#         output_ = torch.full((num_users, num_items), -float('inf'), device=output.device)
#         target_ = torch.full((num_users, num_items), 0., device=target.device)
#         output_[user_idx, item_idx] = output
#         target_[user_idx, item_idx] = target
#     elif cfg['data_mode'] == 'item':
#         output_ = torch.full((num_items, num_users), -float('inf'), device=output.device)
#         target_ = torch.full((num_items, num_users), 0., device=target.device)
#         output_[item_idx, user_idx] = output
#         target_[item_idx, user_idx] = target
#     else:
#         raise ValueError('Not valid data mode')
#
#     return output_, target_
#
#
# def _get_topk_targets(output_, target_, topk):
#     topk = min(topk, target_.size(-1))
#     _, indices = output_.topk(topk, dim=-1)
#     return target_.take_along_dim(indices, dim=-1)
#
#
# def div_no_nan(a, b, na_value=0.):
#     return (a / b).nan_to_num_(nan=na_value, posinf=na_value, neginf=na_value)
#
#
# def DCG(target):
#     batch_size, k = target.shape
#     rank_positions = torch.arange(1, k + 1, dtype=torch.float32, device=target.device).expand(batch_size, -1)
#     return (target / torch.log2(rank_positions + 1)).sum(dim=-1)
#
#
# def MAP(output, target, user, item, topk=10):
#     output_, target_ = _setup_matrix(output, target, user, item)
#     topk_target = _get_topk_targets(output_, target_, topk)
#     precision = torch.cumsum(topk_target, dim=-1) / torch.arange(1, topk + 1, device=output.device).float()
#     m = torch.sum(topk_target, dim=-1)
#     ap = (precision * topk_target).sum(dim=-1) / (m + 1e-10)
#     return ap.mean().item()
#
#
# def Precision(output, target, user, item, topk=10):
#     output_, target_ = _setup_matrix(output, target, user, item)
#     topk_target = _get_topk_targets(output_, target_, topk)
#     relevant_retrieved_items = torch.sum(topk_target, dim=-1)
#     return (relevant_retrieved_items / topk).mean().item()
#
#
# def Recall(output, target, user, item, topk=10):
#     output_, target_ = _setup_matrix(output, target, user, item)
#     topk_target = _get_topk_targets(output_, target_, topk)
#     relevant_items = torch.sum(target_, dim=-1)
#     relevant_retrieved_items = torch.sum(topk_target, dim=-1)
#     return (relevant_retrieved_items / (relevant_items + 1e-10)).mean().item()
#
#
# def F1(output, target, user, item, topk=10):
#     precision = Precision(output, target, user, item, topk)
#     recall = Recall(output, target, user, item, topk)
#     if precision + recall == 0:
#         return 0.0
#     return 2 * (precision * recall) / (precision + recall)
#
#
# def NDCG(output, target, user, item, topk=20):
#     output_, target_ = _setup_matrix(output, target, user, item)
#     sorted_target = _get_topk_targets(output_, target_, topk)
#     ideal_target, _ = target_.topk(topk, dim=-1)
#     ndcg = div_no_nan(DCG(sorted_target), DCG(ideal_target)).mean().item()
#     return ndcg
#
#
# def HR(output, target, user, item, topk=10):
#     output_, target_ = _setup_matrix(output, target, user, item)
#     sorted_target = _get_topk_targets(output_, target_, topk)
#     hr = (sorted_target.float().sum(dim=-1) > 0).float().mean().item()
#     return hr

class Metric:
    def __init__(self, metric_name, best, best_direction, best_metric_name):
        self.metric_name = metric_name
        self.best, self.best_direction, self.best_metric_name = best, best_direction, best_metric_name
        self.metric = self.make_metric(metric_name)

    def make_metric(self, metric_name):
        metric = defaultdict(dict)
        for split in metric_name:
            for m in metric_name[split]:
                if m == 'Loss':
                    metric[split][m] = {'mode': 'batch', 'metric': (lambda input, output: output['loss'].item())}
                elif m == 'Accuracy':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(Accuracy, output['target'], input['target']))}
                elif m == 'MSE':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(MSE, output['target'], input['target']))}
                else:
                    raise ValueError('Not valid metric name')
        return metric

    def add(self, split, input, output):
        for metric_name in self.metric_name[split]:
            if self.metric[split][metric_name]['mode'] == 'full':
                self.metric[split][metric_name]['metric'].add(input, output)
        return

    def evaluate(self, split, mode, input, output, metric_name):
        evaluation = {}
        for metric_name_i in metric_name[split]:
            if self.metric[split][metric_name_i]['mode'] == mode:
                evaluation[metric_name_i] = self.metric[split][metric_name_i]['metric'](input, output)
        return evaluation

    def compare(self, val, if_update):
        if self.best_direction == 'down':
            compared = self.best > val
        elif self.best_direction == 'up':
            compared = self.best < val
        else:
            raise ValueError('Not valid best direction')
        if if_update:
            self.best = val
        return compared

    def load_state_dict(self, state_dict):
        self.best = state_dict['best']
        self.best_metric_name = state_dict['best_metric_name']
        self.best_direction = state_dict['best_direction']
        return

    def state_dict(self):
        return {'best': self.best, 'best_metric_name': self.best_metric_name, 'best_direction': self.best_direction}

