import numpy as np
import torch


class Tokenizer:
    def __init__(self, unk_token='[UNK]', pad_token='[PAD]', cls_token='[CLS]', sep_token='[SEP]', if_train=False):
        super().__init__()
        self.vocab = {}
        self.inv_vocab = {}
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.if_train = if_train
        self.padding_direction = 'right'
        self.special_token = [self.unk_token, self.pad_token, self.cls_token, self.sep_token]
        for i in range(len(self.special_token)):
            self.update(self.special_token[i])

    def update(self, token):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
            self.inv_vocab[self.vocab[token]] = token
        return

    def train(self, if_train):
        self.if_train = if_train
        return

    def __call__(self, input, max_length=None, padding=False, truncation=False, return_tensors='pt'):
        seq_len = [len(input['item'][i]) for i in range(len(input['item']))]
        target_seq_len = [len(input['target_item'][i]) for i in range(len(input['target_item']))]
        print(seq_len)
        print(target_seq_len)
        exit()

        if max_length == 'longest':
            max_length = max(seq_len)

        return

    def tokenize(self, data):
        print(data)
        exit()

        return

    def convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab[self.unk_token])

    def convert_id_to_token(self, index):
        return self.inv_vocab.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)
