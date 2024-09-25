import dataset
import torch
import numpy as np
from config import cfg
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def make_dataset(data_name, transform=True, verbose=True):
    dataset_ = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['AmazonBeauty', 'Gowalla', 'Yelp18']:
        dataset_['train'] = eval('dataset.{}(root=root, split=\'train\')'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split=\'test\')'.format(data_name))
        if transform:
            if cfg['max_length_mode'] == 'longest':
                max_length = dataset_['train'].max_length
                cfg['model']['max_length'] = int(16 * np.ceil(max_length / 16))
            else:
                cfg['model']['max_length'] = 128
            dataset_['train'].transform = dataset.Compose([NegativeSampling(cfg['model']['stats']['num_items'],
                                                                            cfg['model']['num_negatives']),
                                                           Padding(cfg['model']['pad_token'],
                                                                   cfg['model']['max_length'])])
            dataset_['test'].transform = dataset.Compose([NegativeSampling(cfg['model']['stats']['num_items'],
                                                                           cfg['model']['num_negatives']),
                                                          Padding(cfg['model']['pad_token'],
                                                                  cfg['model']['max_length'])])
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset_


def input_collate(input):
    first = input[0]
    batch = {}
    for k, v in first.items():
        if v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in input])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in input]))
            else:
                batch[k] = torch.tensor([f[k] for f in input])
    return batch


def make_data_collate(collate_mode):
    if collate_mode == 'dict':
        return input_collate
    elif collate_mode == 'default':
        return default_collate
    else:
        raise ValueError('Not valid collate mode')


def make_data_loader(dataset, batch_size, num_steps=None, step=0, step_period=1, pin_memory=True,
                     num_workers=0, collate_mode='dict', seed=0, shuffle=True):
    data_loader = {}
    for k in dataset:
        if k == 'train' and num_steps is not None:
            num_samples = batch_size[k] * (num_steps - step) * step_period
            if num_samples > 0:
                generator = torch.Generator()
                generator.manual_seed(seed)
                sampler = torch.utils.data.RandomSampler(dataset[k], replacement=False, num_samples=num_samples,
                                                         generator=generator)
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], sampler=sampler,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
        else:
            if k == 'train':
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], shuffle=shuffle,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
            else:
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], shuffle=False,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
    return data_loader


def process_dataset(dataset):
    processed_dataset = dataset
    cfg['data_size'] = {k: len(processed_dataset[k]) for k in processed_dataset}
    if 'num_epochs' in cfg:
        cfg['num_steps'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size'])) * cfg['num_epochs']
        cfg['eval_period'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size']))
        cfg[cfg['tag']]['optimizer']['num_steps'] = cfg['num_steps']
    return processed_dataset


class Padding(torch.nn.Module):
    def __init__(self, pad_id, max_length):
        super().__init__()
        self.pad_id = pad_id
        self.max_length = max_length

    def forward(self, input):
        if len(input['item_hist']) > self.max_length:
            input['item_hist'] = input['item_hist'][:self.max_length]
        else:
            pad_length = self.max_length - len(input['item_hist'])
            input['item_hist'] = torch.cat(
                [input['item_hist'], input['item_hist'].new_tensor([self.pad_id] * pad_length)])
        return input


class NegativeSampling(torch.nn.Module):
    def __init__(self, num_items, num_negatives):
        super().__init__()
        self.num_items = num_items
        self.num_negatives = num_negatives

    def forward(self, input):
        positives = set(torch.cat([input['item'].view(-1), input['item_hist']]).tolist())
        negatives = set()
        while len(negatives) < self.num_negatives:
            new_negatives = torch.randint(low=0, high=self.num_items, size=(self.num_negatives,))
            negatives.update(set(new_negatives.tolist()) - positives)
        negatives = torch.tensor(list(negatives))
        if len(negatives) > self.num_negatives:
            negatives = negatives[:self.num_negatives]
        negatives = negatives[:self.num_negatives]
        input['item'] = torch.cat([input['item'].view(-1), negatives])
        input['rating'] = torch.cat([input['rating'].view(-1), input['rating'].new_zeros(len(negatives))])
        return input
