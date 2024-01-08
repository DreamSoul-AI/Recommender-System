import numpy as np
import torch


class Tokenizer:
    def __init__(self, unk_token='[UNK]', pad_token='[PAD]', cls_token='[CLS]', sep_token='[SEP]', if_train=False):
        super().__init__()
        self.user_vocab = {}
        self.inv_user_vocab = {}
        self.item_vocab = {}
        self.inv_item_vocab = {}
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.if_train = if_train
        self.padding_direction = 'right'
        self.special_token = [self.unk_token, self.pad_token, self.cls_token, self.sep_token]
        for i in range(len(self.special_token)):
            self.update(self.special_token[i], self.user_vocab, self.inv_user_vocab)
            self.update(self.special_token[i], self.item_vocab, self.inv_item_vocab)

    def update(self, token, vocab, inv_vocab):
        if token not in vocab:
            vocab[token] = len(vocab)
            inv_vocab[vocab[token]] = token
        return

    def train(self, if_train):
        self.if_train = if_train
        return

    def __call__(self, input, max_length=None, padding=False, truncation=False, return_tensors='pt'):
        user = input['user']
        item = input['item']
        target_item = input['target_item']
        seq_len = [len(item[i]) for i in range(len(item))]
        target_seq_len = [len(target_item[i]) for i in range(len(target_item))]
        data = []
        attention_mask = []
        target = []
        for i in range(len(user)):
            user_i, item_i, target_item_i = user[i], item[i], target_item[i]
            user_i, item_i, target_item_i = self.tokenize(user_i, item_i, target_item_i)
            user_i[0] = self.convert_token_to_id(user_i[0], self.user_vocab)
            item_i = [self.convert_token_to_id(item_i[j], self.item_vocab) for j in range(len(item_i))]
            target_item_i = [self.convert_token_to_id(target_item_i[j], self.item_vocab) for j in
                             range(len(target_item_i))]
            print(user_i)
            print(item_i)
            print(target_item_i)
            exit()
        if max_length == 'longest':
            max_length = max(seq_len)

        return

    def tokenize(self, user, item, target_item):
        if self.if_train:
            self.update(user[0], self.user_vocab, self.inv_user_vocab)
        for j in range(len(item)):
            item_j = item[j]
            target_item_j = target_item[j]
            if self.if_train:
                self.update(item_j, self.item_vocab, self.inv_item_vocab)
                self.update(target_item_j, self.item_vocab, self.inv_item_vocab)
        return user, item, target_item

    def convert_token_to_id(self, token, vocab):
        return vocab.get(token, vocab[self.unk_token])

    def convert_id_to_token(self, index, inv_vocab):
        return inv_vocab.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)
