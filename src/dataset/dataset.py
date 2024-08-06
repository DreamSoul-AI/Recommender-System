import dataset
import torch
import numpy as np
from datasets import Dataset
from collections import defaultdict
from config import cfg
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from functools import partial


def make_dataset(data_name, target_mode=None, verbose=True):
    dataset_ = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['ML100K', 'ML1M', 'ML10M', 'ML20M']:
        dataset_['train'] = eval(
            'dataset.{}(root=root, split=\'train\', target_mode=target_mode)'.format(data_name))
        dataset_['test'] = eval(
            'dataset.{}(root=root, split=\'test\', target_mode=target_mode)'.format(data_name))
    elif data_name in ['AmazonBeauty']:
        dataset_['train'] = eval('dataset.{}(root=root, split=\'train\')'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split=\'test\')'.format(data_name))
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
        if cfg['target_mode'] == 'implicit':
            processed_dataset[split].set_transform(NagativeSampling(tokenizer))

    cfg['data_size'] = {'train': len(dataset['train']), 'test': len(dataset['test'])}
    cfg['num_users'], cfg['num_items'] = dataset['train'].num_users, dataset['train'].num_items
    if 'num_epochs' in cfg:
        cfg['num_steps'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size'])) * cfg['num_epochs']
        cfg['eval_period'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size']))
        cfg[cfg['tag']]['optimizer']['num_steps'] = cfg['num_steps']

    cfg['model']['user_vocab_size'] = len(tokenizer.user_vocab)
    cfg['model']['item_vocab_size'] = len(tokenizer.item_vocab)
    return processed_dataset


class NagativeSampling(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        item_id = set(tokenizer.inv_item_vocab.keys())
        special_id = set([tokenizer.convert_token_to_id(x, tokenizer.item_vocab) for x in tokenizer.special_token])
        self.data_id = item_id - special_id
        self.pad_id = tokenizer.convert_token_to_id(tokenizer.pad_token, tokenizer.item_vocab)
        self.negative_ratio = 1.0

    def make_negative_sample(self, item, positive_seq_len, negative_seq_len, pad_len_i):
        positive_item = item[:positive_seq_len]
        negative_item_pool = torch.tensor(list(self.data_id - set(positive_item)))
        negative_item = negative_item_pool[torch.randperm(len(negative_item_pool))[:negative_seq_len]].tolist()
        item = positive_item + negative_item + [self.pad_id] * pad_len_i
        rating = [1.] * len(positive_item) + [0.] * len(negative_item) + [0.] * pad_len_i
        attention_mask = [True] * len(positive_item) + [True] * len(negative_item) + [False] * pad_len_i
        return item, rating, attention_mask

    def forward(self, input):
        max_seq_len = torch.tensor(input['attention_mask']).size(1)
        positive_seq_len = torch.tensor(input['attention_mask']).sum(dim=1)
        negative_seq_len = torch.round(self.negative_ratio * positive_seq_len).long()
        new_max_seq_len = max(positive_seq_len.max() + negative_seq_len.max(), max_seq_len)

        target_max_seq_len = torch.tensor(input['target_attention_mask']).size(1)
        target_positive_seq_len = torch.tensor(input['target_attention_mask']).sum(dim=1)
        target_negative_seq_len = torch.round(self.negative_ratio * target_positive_seq_len).long()
        target_new_max_seq_len = max(target_positive_seq_len.max() + target_negative_seq_len.max(), target_max_seq_len)
        for i in range(len(input['user'])):
            pad_len_i = new_max_seq_len - (positive_seq_len[i] + negative_seq_len[i])
            target_pad_len_i = target_new_max_seq_len - (target_positive_seq_len[i] + target_negative_seq_len[i])
            input['item'][i], input['rating'][i], input['attention_mask'][i] = self.make_negative_sample(
                input['item'][i], positive_seq_len[i], negative_seq_len[i], pad_len_i)
            input['target_item'][i], input['target_rating'][i], input['target_attention_mask'][i] = (
                self.make_negative_sample(input['target_item'][i], target_positive_seq_len[i],
                                          target_negative_seq_len[i], target_pad_len_i))

        return input
