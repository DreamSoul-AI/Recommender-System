import os
import torch
from torchvision import transforms
from config import cfg
from dataset import make_dataset, make_data_loader, process_dataset, Compose
from module import save, Stats, makedir_exist_ok, process_control

if __name__ == "__main__":
    stats_path = os.path.join('output', 'stats')
    dim = 1
    # data_names = ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'Douban', 'Amazon']
    data_names = ['ML100K']
    target_modes = ['explicit', 'implicit']
    cfg['seed'] = 0
    cfg['tag'] = 'make_dataset'
    process_control()
    for data_name in data_names:
        for target_mode in target_modes:
            dataset = make_dataset(data_name, target_mode)
            stats = {'m': dataset['train'].num_users, 'n': dataset['train'].num_items}
            rating = dataset['train'].data['rating']
            stats['min'] = min(min(rating_i) for rating_i in rating if len(rating_i) > 0)
            stats['max'] = max(max(rating_i) for rating_i in rating if len(rating_i) > 0)
            print(data_name, target_mode, stats)
            makedir_exist_ok(stats_path)
            save(stats, os.path.join(stats_path, '{}_{}'.format(data_name, target_mode)))
