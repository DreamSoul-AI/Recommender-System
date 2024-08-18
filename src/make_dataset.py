import os
import torch
from config import cfg
from dataset import make_dataset
from module import save, makedir_exist_ok, process_control

if __name__ == "__main__":
    stats_path = os.path.join('output', 'stats')
    dim = 1
    data_names = ['AmazonBeauty']
    cfg['seed'] = 0
    cfg['tag'] = 'make_dataset'
    cfg['remake_stats'] = True
    process_control()
    for data_name in data_names:
        dataset = make_dataset(data_name, transform=False)
        stats = {'num_users': dataset['train'].num_users, 'num_items': dataset['train'].num_items,
                 'num_ratings': {'train': dataset['train'].num_ratings, 'test': dataset['test'].num_ratings}}
        rating = dataset['train'].data['train'].data
        stats['min'] = min(rating)
        stats['max'] = max(rating)
        print(data_name, stats)
        makedir_exist_ok(stats_path)
        save(stats, os.path.join(stats_path, '{}'.format(data_name)))
