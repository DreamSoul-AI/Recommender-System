import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
from .utils import evaluate_metrics


def make_metric(split, **kwargs):
    data_name = kwargs['data_name']
    metric_name = {k: [] for k in split}
    if data_name in ['AmazonBeauty']:
        best_direction = 'up'
        best_metric_name = 'Recall(k=20)'
        for k in metric_name:
            metric_name[k].extend(['Loss'])
            if kwargs['run_mode'] == 'train' and k == 'test':
                metric_names = ['Recall(k=20)', 'NDCG(k=20)']
                metric_name['test'].extend(metric_names)
            elif kwargs['run_mode'] == 'test' and k == 'test':
                # metric_names = ['F1(k=20)', 'Recall(k=20)', 'nRecall(k=50)',
                #                 'Precision(k=20)', 'F1(k=10)', 'DCG(k=30)', 'NDCG(k=20)',
                #                 'NDCG(k=50)', 'MRR(k=30)', 'HitRate(k=20)',
                #                 'HitRate(k=50)', 'MAP(k=10)']
                # metric_names = ['NDCG(k=20)', 'Recall(k=20)']
                metric_names = ['Recall(k=20)', 'Recall(k=50)', 'NDCG(k=20)', 'NDCG(k=50)', 'HitRate(k=20)',
                                'HitRate(k=50)']
                metric_name['test'].extend(metric_names)
    else:
        raise ValueError('Not valid data name')
    metric = Metric(metric_name, best_direction, best_metric_name,
                    num_users=kwargs['num_users'], num_items=kwargs['num_items'],
                    train_user2items=kwargs['train_user2items'], valid_user2items=kwargs['valid_user2items'])
    return metric


class BaseMetric:
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Loss(BaseMetric):
    def __call__(self, loss):
        with torch.no_grad():
            loss = loss.item()
        return loss


class Accuracy(BaseMetric):
    def __init__(self, topk=1):
        super().__init__()
        self.topk = topk

    def __call__(self, pred, target):
        with torch.no_grad():
            if target.dtype != torch.int64:
                target = (target.topk(1, -1, True, True)[1]).view(-1)
            batch_size = torch.numel(target)
            if pred.dtype != torch.int64:
                pred_k = pred.topk(self.topk, -1, True, True)[1]
                correct_k = pred_k.eq(target.unsqueeze(-1).expand_as(pred_k)).float().sum()
            else:
                correct_k = pred.eq(target).float().sum()
            acc = (correct_k * (100.0 / batch_size)).item()
        return acc


class MSE(BaseMetric):
    def __call__(self, pred, target):
        with torch.no_grad():
            mse = F.mse_loss(pred, target).item()
        return mse


class RMSE(BaseMetric):
    def __call__(self, pred, target):
        with torch.no_grad():
            rmse = F.mse_loss(pred, target).sqrt().item()
        return rmse


class RS:
    def __init__(self, num_users, num_items, train_user2items, valid_user2items):
        self.metric_names = []
        self.num_users = num_users
        self.num_items = num_items
        self.train_user2items = train_user2items
        self.valid_user2items = valid_user2items
        self.query_indices = list(range(self.num_users))

    def add_metric(self, metric_name):
        self.metric_names.append(metric_name)
        return

    def __call__(self, input, output):
        with torch.no_grad():
            if len(self.metric_names) > 0:
                user_embedding = output['user_embedding'].cpu().numpy().astype(np.float64)
                item_embedding = output['item_embedding'].cpu().numpy().astype(np.float64)
                rs = evaluate_metrics(user_embedding, item_embedding,
                                      self.train_user2items, self.valid_user2items,
                                      self.query_indices, self.metric_names)
            else:
                rs = {}
        return rs


# def evaluate(self, train_generator, valid_generator):
#     logging.info("Start evaluation...")
#     self.eval()  # set to evaluation mode
#     with torch.no_grad():
#         user_vecs = []
#         item_vecs = []
#         for user_batch in valid_generator.user_loader:
#             user_vec = self.user_tower(user_batch)
#             user_vecs.extend(user_vec.data.cpu().numpy())
#         for item_batch in valid_generator.item_loader:
#             item_vec = self.item_tower(item_batch)
#             item_vecs.extend(item_vec.data.cpu().numpy())
#         user_vecs = np.array(user_vecs, np.float64)
#         item_vecs = np.array(item_vecs, np.float64)
#         val_logs = evaluate_metrics(user_vecs,
#                                     item_vecs,
#                                     train_generator.user2items_dict,
#                                     valid_generator.user2items_dict,
#                                     valid_generator.query_indexes,
#                                     self._validation_metrics)
#         return val_logs


class Metric:
    def __init__(self, metric_name, best_direction, best_metric_name, **kwargs):
        self.rs_metric_names = ['F1', 'Recall', 'nRecall', 'Precision', 'F1', 'DCG', 'NDCG', 'MRR', 'HitRate',
                                'HitRate', 'MAP']
        self.rs = RS(kwargs['num_users'], kwargs['num_items'], kwargs['train_user2items'], kwargs['valid_user2items'])
        self.metric_name = metric_name
        self.best_direction, self.best_metric_name = best_direction, best_metric_name
        self.metric, self.mode, self.mode_keys = self.make_metric(metric_name)
        self.full_mode_keys = self.make_full_mode(self.mode, self.mode_keys)
        self.reset()

    def make_metric(self, metric_name):
        metric = {}
        mode = {}
        mode_keys = {}
        for split in metric_name:
            metric[split] = {}
            mode[split] = {}
            mode_keys[split] = {}
            for metric_name_i in metric_name[split]:
                mode_keys[split][metric_name_i] = {'input': set(), 'output': set()}
                if metric_name_i in ['Loss']:
                    metric[split][metric_name_i] = eval('{}()'.format(metric_name_i))
                    mode[split][metric_name_i] = 'batch'
                    mode_keys[split][metric_name_i]['output'].add('loss')
                elif metric_name_i in ['Accuracy', 'MSE']:
                    metric[split][metric_name_i] = eval('{}()'.format(metric_name_i))
                    mode[split][metric_name_i] = 'batch'
                    mode_keys[split][metric_name_i]['input'].add('target')
                    mode_keys[split][metric_name_i]['output'].add('pred')
                elif any(rs_metric_name in metric_name_i for rs_metric_name in self.rs_metric_names):
                    self.rs.add_metric(metric_name_i)
                    mode[split][metric_name_i] = 'full'
                    mode_keys[split][metric_name_i]['output'].update(['user_embedding', 'item_embedding'])
                else:
                    raise ValueError('Not valid metric name')
        return metric, mode, mode_keys

    def make_init_best(self):
        if self.best_direction == 'up':
            init_best = -float('inf')
        elif self.best_direction == 'down':
            init_best = float('inf')
        else:
            raise ValueError('Not valid best direction')
        return init_best

    def make_full_mode(self, mode, mode_keys):
        full_mode_keys = {}
        for split in mode:
            full_mode_keys[split] = {'input': set(), 'output': set()}
            for metric_name_i in mode[split]:
                if mode[split][metric_name_i] == 'full':
                    full_mode_keys[split]['input'].update(mode_keys[split][metric_name_i]['input'])
                    full_mode_keys[split]['output'].update(mode_keys[split][metric_name_i]['output'])
        return full_mode_keys

    def add(self, split, input, output):
        with torch.no_grad():
            for key in self.full_mode_keys[split]['input']:
                if key not in self.buffer['input']:
                    self.buffer['input'][key] = input[key]
                else:
                    self.buffer['input'][key] = torch.cat([self.buffer['input'][key], input[key]], dim=0)
            for key in self.full_mode_keys[split]['output']:
                if key not in self.buffer['output']:
                    self.buffer['output'][key] = output[key]
                else:
                    self.buffer['output'][key] = torch.cat([self.buffer['output'][key], output[key]], dim=0)
        return

    def evaluate(self, split, mode, input=None, output=None, metric_name=None):
        metric_name = self.metric_name if metric_name is None else metric_name
        evaluation = {}
        if mode == 'batch':
            for metric_name_i in metric_name[split]:
                if self.mode[split][metric_name_i] == mode:
                    input_ = {key: input[key] for key in self.mode_keys[split][metric_name_i]['input']}
                    output_ = {key: output[key] for key in self.mode_keys[split][metric_name_i]['output']}
                    evaluation[metric_name_i] = self.metric[split][metric_name_i](**input_, **output_)
        elif mode == 'full':
            for metric_name_i in metric_name[split]:
                if self.mode[split][metric_name_i] == mode:
                    input_ = {key: self.buffer['input'][key] for key in self.mode_keys[split][metric_name_i]['input']}
                    output_ = {key: self.buffer['output'][key] for key in
                               self.mode_keys[split][metric_name_i]['output']}
                    if any(rs_metric_name in metric_name_i for rs_metric_name in self.rs_metric_names):
                        if metric_name_i == self.rs.metric_names[0]:
                            rs_metric = self.rs(input_, output_)
                            for rs_metric_name_i in rs_metric:
                                evaluation[rs_metric_name_i] = rs_metric[rs_metric_name_i]
                    else:
                        evaluation[metric_name_i] = self.metric[split][metric_name_i](**input_, **output_)
            self.reset_buffer()
        else:
            raise ValueError('Not valid mode')
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

    def reset(self):
        self.reset_best()
        self.reset_buffer()
        return

    def reset_best(self):
        self.best = self.make_init_best()
        return

    def reset_buffer(self):
        self.buffer = {'input': {}, 'output': {}}
        return

    def load_state_dict(self, state_dict):
        self.best_metric_name = state_dict['best_metric_name']
        self.best_direction = state_dict['best_direction']
        self.reset_best()
        return

    def state_dict(self):
        return {'best_metric_name': self.best_metric_name, 'best_direction': self.best_direction}
