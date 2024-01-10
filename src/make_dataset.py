import os
import torch
from config import cfg
from dataset import make_dataset
from module import save, makedir_exist_ok, process_control

if __name__ == "__main__":
    stats_path = os.path.join('res', 'stats')
    cfg['seed'] = 0

    # data_names = ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'Douban', 'Amazon']
    data_names = ['ML100K']
    target_modes = ['explicit', 'implicit']
    with torch.no_grad():
        for data_name in data_names:
            for target_mode in target_modes:
                cfg['control']['data_name'] = data_name
                cfg['control']['target_mode'] = target_mode
                process_control()
                dataset = make_dataset(cfg['data_name'], verbose=False)
                stats = {'m': dataset['train'].num_users, 'n': dataset['train'].num_items}
                print(data_name, target_mode, stats)
                makedir_exist_ok(stats_path)
                save(stats, os.path.join(stats_path, '{}_{}'.format(data_name, target_mode)))
