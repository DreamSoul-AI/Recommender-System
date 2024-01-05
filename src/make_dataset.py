import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import fetch_dataset
from module import save, process_control, makedir_exist_ok

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

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
                dataset = fetch_dataset(cfg['data_name'], verbose=False)
                stats = {'m': dataset['train'].num_users, 'n': dataset['train'].num_items}
                print(data_name, target_mode, stats)
                makedir_exist_ok(stats_path)
                save(stats, os.path.join(stats_path, '{}_{}'.format(data_name, target_mode)))
