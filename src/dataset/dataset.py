import dataset
import torch
import numpy as np
from datasets import Dataset
from collections import defaultdict
from config import cfg
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from functools import partial


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


def process_dataset(dataset, tokenizer):
    cfg['max_length'] = {'train': {'item': max([len(x) for x in dataset['train'].data['item']]),
                                   'target_item': max([len(x) for x in dataset['train'].data['target_item']])},
                         'test': {'item': max([len(x) for x in dataset['test'].data['item']]),
                                  'target_item': max([len(x) for x in dataset['test'].data['target_item']])}}

    def preprocess_function(max_length, examples):
        model_inputs = tokenizer(examples, max_length=max_length, padding=True, truncation=False,
                                 return_tensors='pt')
        return model_inputs

    processed_dataset = {}
    for split in dataset:
        data = defaultdict(list)
        for k in dataset[split].data:
            data[k].extend(dataset[split].data[k])
        processed_dataset[split] = Dataset.from_dict(data)
        preprocess_function_ = partial(preprocess_function, cfg['max_length'][split])
        processed_dataset[split] = processed_dataset[split].map(
            preprocess_function_,
            batched=True,
            num_proc=1,
            load_from_cache_file=False,
            desc="Preprocess dataset",
            batch_size=50,
        )

    cfg['data_size'] = {'train': len(dataset['train']), 'test': len(dataset['test'])}
    cfg['num_users'], cfg['num_items'] = dataset['train'].num_users, dataset['train'].num_items
    if 'num_epochs' in cfg:
        cfg['num_steps'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size'])) * cfg['num_epochs']
        cfg['eval_period'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size']))

    cfg['model']['user_vocab_size'] = len(tokenizer.user_vocab)
    cfg['model']['item_vocab_size'] = len(tokenizer.item_vocab)
    return processed_dataset
