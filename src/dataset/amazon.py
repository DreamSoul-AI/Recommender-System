import numpy as np
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from .utils import download_url
from scipy.sparse import coo_matrix
from datasets import load_dataset
from module import check_exists, makedir_exist_ok, save, load


class AmazonBeauty(Dataset):
    data_name = 'AmazonBeauty'
    hf_data_name = 'reczoo/AmazonBeauty_m1'

    def __init__(self, root, split, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        # if not check_exists(self.processed_folder):
        self.process()
        self.data, self.meta = load(os.path.join(self.processed_folder, '{}'.format(self.split)), mode='pickle')
        self.train_data_csr = self.data['train'].tocsr()

    def __getitem__(self, index):
        user, item, target = self.data[self.split].row, self.data[self.split].col, self.data[self.split].data
        item_hist = self.train_data_csr[user]
        print(user, item, target)
        print(item_hist)
        exit()

        return input

    def __len__(self):
        return len(self.data)

    @property
    def num_users(self):
        return self.meta['num_users']

    @property
    def num_items(self):
        return self.meta['num_items']

    @property
    def num_ratings(self):
        return self.meta['num_ratings'][self.split]

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        dataset = self.make_data()
        save(dataset, os.path.join(self.processed_folder), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        load_dataset(self.hf_data_name, cache_dir=self.raw_folder)
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split)
        return fmt_str

    def make_data(self):
        dataset = load_dataset(self.hf_data_name, cache_dir=self.raw_folder)
        dataset = self.parse_dataset(dataset)
        meta = {'num_users': dataset['train'].shape[0], 'num_items': dataset['train'].shape[1],
                'num_ratings': {'train': dataset['train'].nnz, 'test': dataset['test'].nnz}}
        return dataset, meta

    def parse_dataset(self, dataset):
        user_token_to_index = {}
        item_token_to_index = {}
        user_counter = 0
        item_counter = 0

        train_users = []
        train_items = []
        for row in dataset['train']:
            tokens = list(map(int, row['text'].split()))
            user_token = tokens[0]
            item_tokens = tokens[1:]

            if user_token not in user_token_to_index:
                user_token_to_index[user_token] = user_counter
                user_counter += 1
            user_index = user_token_to_index[user_token]

            for item_token in item_tokens:
                if item_token not in item_token_to_index:
                    item_token_to_index[item_token] = item_counter
                    item_counter += 1
                item_index = item_token_to_index[item_token]

                train_users.append(user_index)
                train_items.append(item_index)

        test_users = []
        test_items = []
        for row in dataset['test']:
            tokens = list(map(int, row['text'].split()))
            user_token = tokens[0]
            item_tokens = tokens[1:]

            user_index = user_token_to_index[user_token]

            for item_token in item_tokens:
                item_index = item_token_to_index[item_token]
                test_users.append(user_index)
                test_items.append(item_index)

        num_users = len(user_token_to_index)
        num_items = len(item_token_to_index)
        dataset['train'] = coo_matrix((np.ones(len(train_users)), (train_users, train_items)),
                                      shape=(num_users, num_items))
        dataset['test'] = coo_matrix((np.ones(len(test_users)), (test_users, test_items)),
                                     shape=(num_users, num_items))
        return dataset
