import numpy as np
import torch


class Tokenizer:
    def __init__(self, pad_token='[PAD]', if_train=False):
        super().__init__()
        self.user_vocab = {}
        self.inv_user_vocab = {}
        self.item_vocab = {}
        self.inv_item_vocab = {}
        self.pad_token = pad_token
        self.if_train = if_train
        self.padding_direction = 'right'
        self.special_token = [self.pad_token]
        for i in range(len(self.special_token)):
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
        rating = input['rating']
        target_item = input['target_item']
        target_rating = input['target_rating']
        seq_len = [len(item[i]) for i in range(len(item))]
        target_seq_len = [len(target_item[i]) for i in range(len(target_item))]
        if max_length['item'] == 'longest':
            max_length['item'] = max(seq_len)
        if max_length['target_item'] == 'longest':
            max_length['target_item'] = max(seq_len)
        user_ = []
        item_ = []
        rating_ = []
        target_item_ = []
        target_rating_ = []
        attention_mask = []
        target_attention_mask = []
        for i in range(len(user)):
            user_i, item_i, rating_i, target_item_i, target_rating_i = (user[i], item[i], rating[i],
                                                                        target_item[i], target_rating[i])
            user_i, item_i, target_item_i = self.tokenize(user_i, item_i, target_item_i)
            user_i[0] = self.convert_token_to_id(user_i[0], self.user_vocab)
            item_i = [self.convert_token_to_id(item_i[j], self.item_vocab) for j in range(len(item_i))]
            target_item_i = [self.convert_token_to_id(target_item_i[j], self.item_vocab) for j in
                             range(len(target_item_i))]
            if truncation and max_length is not None:
                if max_length['item'] < seq_len[i]:
                    item_i = item_i[:max_length['item']]
                    rating_i = rating_i[:max_length['item']]
                    seq_len[i] = len(item_i)
                if max_length['target_item'] < target_seq_len[i]:
                    target_item_i = target_item_i[:max_length['target_item']]
                    target_rating_i = target_rating_i[:max_length['target_item']]
                    target_seq_len[i] = len(target_item_i)
            if padding and max_length is not None and max_length['item'] > seq_len[i]:
                pad_width = max_length['item'] - seq_len[i]
                pad_value = self.convert_token_to_id(self.pad_token, self.item_vocab)
                if self.padding_direction == 'right':
                    item_i = np.pad(item_i, (0, pad_width), mode='constant', constant_values=pad_value).tolist()
                    rating_i = np.pad(rating_i, (0, pad_width), mode='constant', constant_values=0).tolist()
                    attention_mask_i = [1] * seq_len[i] + [0] * pad_width
                elif self.padding_direction == 'left':
                    item_i = np.pad(item_i, (pad_width, 0), mode='constant', constant_values=pad_value).tolist()
                    rating_i = np.pad(rating_i, (pad_width, 0), mode='constant', constant_values=0).tolist()
                    attention_mask_i = [0] * pad_width + [1] * seq_len[i]
                else:
                    raise ValueError('Not valid padding direction')
            else:
                attention_mask_i = [1] * seq_len[i]
            if padding and max_length is not None and max_length['target_item'] > target_seq_len[i]:
                target_pad_width = max_length['target_item'] - target_seq_len[i]
                target_pad_value = self.convert_token_to_id(self.pad_token, self.item_vocab)
                if self.padding_direction == 'right':
                    target_item_i = np.pad(target_item_i, (0, target_pad_width), mode='constant',
                                           constant_values=target_pad_value).tolist()
                    target_rating_i = np.pad(target_rating_i, (0, target_pad_width), mode='constant',
                                             constant_values=0).tolist()
                    target_attention_mask_i = [1] * target_seq_len[i] + [0] * target_pad_width
                elif self.padding_direction == 'left':
                    target_item_i = np.pad(target_item_i, (target_pad_width, 0), mode='constant',
                                           constant_values=target_pad_value).tolist()
                    target_rating_i = np.pad(target_rating_i, (target_pad_width, 0), mode='constant',
                                             constant_values=0).tolist()
                    target_attention_mask_i = [0] * target_pad_width + [1] * target_seq_len[i]
                else:
                    raise ValueError('Not valid padding direction')
            else:
                target_attention_mask_i = [1] * target_seq_len[i]
            user_.append(user_i)
            item_.append(item_i)
            rating_.append(rating_i)
            attention_mask.append(attention_mask_i)
            target_item_.append(target_item_i)
            target_rating_.append(target_rating_i)
            target_attention_mask.append(target_attention_mask_i)
        if return_tensors == 'np':
            user = np.array(user_, dtype=np.int64)
            item = np.array(item_, dtype=np.int64)
            rating = torch.tensor(rating_, dtype=torch.float32)
            attention_mask = np.array(attention_mask, dtype=bool)
            target_item = np.array(target_item_, dtype=np.int64)
            target_rating = np.array(target_rating_, dtype=np.float32)
            target_attention_mask = np.array(target_attention_mask, dtype=bool)
        elif return_tensors == 'pt':
            user = torch.tensor(user_, dtype=torch.long)
            item = torch.tensor(item_, dtype=torch.long)
            rating = torch.tensor(rating_, dtype=torch.float32)
            attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
            target_item = torch.tensor(target_item_, dtype=torch.long)
            target_rating = torch.tensor(target_rating_, dtype=torch.float32)
            target_attention_mask = torch.tensor(target_attention_mask, dtype=torch.bool)
        else:
            raise ValueError('Not valid return tensor')
        output = {'user': user, 'item': item, 'rating': rating, 'attention_mask': attention_mask,
                  'target_item': target_item, 'target_rating': target_rating,
                  'target_attention_mask': target_attention_mask}
        return output

    def tokenize(self, user, item, target_item):
        if self.if_train:
            self.update(user[0], self.user_vocab, self.inv_user_vocab)
        for j in range(len(item)):
            item_j = item[j]
            if self.if_train:
                self.update(item_j, self.item_vocab, self.inv_item_vocab)
        for j in range(len(target_item)):
            target_item_j = target_item[j]
            if self.if_train:
                self.update(target_item_j, self.item_vocab, self.inv_item_vocab)
        return user, item, target_item

    def convert_token_to_id(self, token, vocab):
        return vocab.get(token, vocab[self.pad_token])

    def convert_id_to_token(self, index, inv_vocab):
        return inv_vocab.get(index, self.pad_token)

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)
