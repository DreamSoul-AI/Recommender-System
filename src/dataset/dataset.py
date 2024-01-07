import dataset
import torch
import numpy as np
from datasets import Dataset
from collections import defaultdict
from config import cfg
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def make_dataset(data_name, model_name=None, verbose=True):
    model_name = cfg['model_name'] if model_name is None else model_name
    dataset_ = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'Douban', 'Amazon']:
        dataset_['train'] = eval(
            'dataset.{}(root=root, split=\'train\', target_mode=cfg["target_mode"])'.format(data_name))
        dataset_['test'] = eval(
            'dataset.{}(root=root, split=\'test\', target_mode=cfg["target_mode"])'.format(data_name))
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset_


def input_collate(batch):
    if isinstance(batch[0], dict):
        return {key: [b[key] for b in batch] for key in batch[0]}
    else:
        return default_collate(batch)


def make_data_collate(collate_mode):
    if collate_mode == 'dict':
        return input_collate
    elif collate_mode == 'default':
        return default_collate
    else:
        raise ValueError('Not valid collate mode')


def make_data_loader(dataset, tag, batch_size=None, shuffle=None, sampler=None):
    data_loader = {}
    for k in dataset:
        _batch_size = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        if sampler is None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, sampler=sampler[k],
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
    return data_loader


def process_dataset(dataset, tokenizer):
    max_length = cfg['max_length']

    def preprocess_function(examples):
        model_inputs = tokenizer(examples, max_length=max_length, padding=True, truncation=True,
                                 return_tensors='pt')
        return model_inputs

    processed_dataset = {}
    for split in dataset:
        data = defaultdict(list)
        for k in dataset[split].data:
            data[k].extend(dataset[split].data[k])
        processed_dataset[split] = Dataset.from_dict(data)
        processed_dataset[split] = processed_dataset[split].map(
            preprocess_function,
            batched=True,
            num_proc=1,
            load_from_cache_file=False,
            desc="Preprocess dataset",
            batch_size=50,
        )

    cfg['data_size'] = {'train': len(dataset['train']), 'test': len(dataset['test'])}
    cfg['num_users'], cfg['num_items'] = dataset['train'].num_users, dataset['train'].num_items
    return
