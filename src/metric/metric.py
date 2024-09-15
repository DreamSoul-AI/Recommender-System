import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
from collections import defaultdict
from .utils import evaluate_metrics


def make_metric(split, **kwargs):
    data_name = kwargs['data_name']
    metric_name = {k: [] for k in split}
    if data_name in ['AmazonBeauty']:
        best = float('inf')
        best_direction = 'down'
        best_metric_name = 'Loss'
        for k in metric_name:
            metric_name[k].extend(['Loss'])
            if k == 'test':
                # metric_names = ['F1(k=20)', 'Recall(k=20)', 'nRecall(k=50)',
                #                 'Precision(k=20)', 'F1(k=10)', 'DCG(k=30)', 'NDCG(k=20)',
                #                 'NDCG(k=50)', 'MRR(k=30)', 'HitRate(k=20)',
                #                 'HitRate(k=50)', 'MAP(k=10)']
                # metric_names = ['F1(k=20)']
                metric_names = ['NDCG(k=20)', 'HitRate(k=20)']
                metric_name['test'].extend(metric_names)
    else:
        raise ValueError('Not valid data name')
    pad_token = kwargs['pad_token']
    metric = Metric(metric_name, best, best_direction, best_metric_name, pad_token=pad_token)
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


class RMSE:
    def __init__(self):
        self.reset()

    def reset(self):
        self.se = 0
        self.count = 0
        return

    def add(self, input, output):
        with torch.no_grad():
            self.se += F.mse_loss(output['target_rating'], input['target_rating'], reduction='sum')
            self.count += output['target_rating'].numel()
        return

    def __call__(self, input, output):
        with torch.no_grad():
            rmse = ((self.se / self.count) ** 0.5).item()
        self.reset()
        return rmse


class RS:
    def __init__(self, pad_token):
        self.metric_names = []
        self.pad_token = pad_token
        self.reset()

    def reset(self):
        self.user_embs = None
        self.item_embs = None
        self.train_user2items = defaultdict(list)
        self.valid_user2items = defaultdict(list)
        self.query_indices = None
        return

    def add_metric(self, metric_name):
        self.metric_names.append(metric_name)
        return

    def add(self, input, output):
        with torch.no_grad():
            if output['user_embedding'] is not None:
                if self.user_embs is None:
                    self.user_embs = output['user_embedding']
                else:
                    self.user_embs = torch.cat([self.user_embs, output['user_embedding']])
            if output['item_embedding'] is not None:
                if self.item_embs is None:
                    self.item_embs = output['item_embedding'][:, 0]
                else:
                    self.item_embs = torch.cat([self.item_embs, output['item_embedding'][:, 0]])
            if self.query_indices is None:
                self.query_indices = input['user']
            else:
                self.query_indices = torch.cat([self.query_indices, input['user']])
            for i in range(len(input['user'])):
                user_i = input['user'][i].item()
                item_hist_i = input['item_hist'][i]
                item_hist_i = item_hist_i[item_hist_i != self.pad_token].tolist()
                self.train_user2items[user_i].extend(item_hist_i)
                self.valid_user2items[user_i].append(input['item'][i][0].item())
        return

    def __call__(self, input, output):
        with torch.no_grad():
            if len(self.metric_names) > 0:
                rs = evaluate_metrics(self.user_embs.cpu().numpy(), self.item_embs.cpu().numpy(),
                                      self.train_user2items, self.valid_user2items,
                                      self.query_indices.cpu().numpy(), self.metric_names)
            else:
                rs = {}
        self.reset()
        return rs


class Metric:
    def __init__(self, metric_name, best, best_direction, best_metric_name, **kwargs):
        self.metric_name = metric_name
        self.best, self.best_direction, self.best_metric_name = best, best_direction, best_metric_name
        self.rs_metric_names = ['F1', 'Recall', 'nRecall', 'Precision', 'F1', 'DCG', 'NDCG', 'MRR', 'HitRate',
                                'HitRate', 'MAP']
        self.rs = RS(kwargs['pad_token'])
        self.metric = self.make_metric(metric_name, **kwargs)

    def make_metric(self, metric_name, **kwargs):
        metric = defaultdict(dict)
        for split in metric_name:
            for m in metric_name[split]:
                if m == 'Loss':
                    metric[split][m] = {'mode': 'batch', 'metric': (lambda input, output: output['loss'].item())}
                elif m == 'Accuracy':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (lambda input, output: Accuracy(output['target'], input['target']))}
                elif m == 'MSE':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (lambda input, output: MSE(output['target'], input['target']))}
                elif m == 'RMSE':
                    metric[split][m] = {'mode': 'full', 'metric': RMSE()}
                elif any(rs_metric_name in m for rs_metric_name in self.rs_metric_names):
                    self.rs.add_metric(m)
                    metric[split][m] = {'mode': 'full', 'metric': self.rs}
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
                if any(rs_metric_name in metric_name_i for rs_metric_name in self.rs_metric_names):
                    if metric_name_i == self.metric[split][metric_name_i]['metric'].metric_names[0]:
                        print(metric_name_i)
                        rs_metric = self.metric[split][metric_name_i]['metric'](input, output)
                        for rs_metric_name_i in rs_metric:
                            evaluation[rs_metric_name_i] = rs_metric[rs_metric_name_i]
                else:
                    evaluation[metric_name_i] = self.metric[split][metric_name_i]['metric'](input, output)
        return evaluation

    def compare(self, val, if_update):
        if self.best_direction == 'down':
            compared = self.best > val
        elif self.best_direction == 'up':
            compared = self.best < val
        else:
            raise ValueError('Not valid best direction')
        if compared and if_update:
            self.best = val
        return compared

    def load_state_dict(self, state_dict):
        self.best = state_dict['best']
        self.best_metric_name = state_dict['best_metric_name']
        self.best_direction = state_dict['best_direction']
        return

    def state_dict(self):
        return {'best': self.best, 'best_metric_name': self.best_metric_name, 'best_direction': self.best_direction}
