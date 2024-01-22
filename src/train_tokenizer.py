import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, process_dataset
from model import make_tokenizer
from module import save, process_control

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['tag'] = '_'.join([x for x in tag_list if x])
        process_control()
        print('Experiment: {}'.format(cfg['tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    cfg['path'] = os.path.join('output', 'exp')
    cfg['tokenizer_path'] = os.path.join(cfg['path'], 'tokenizer')
    tokenizer = make_tokenizer()
    dataset = make_dataset(cfg['data_name'], cfg['target_mode'])
    tokenizer.train(True)
    dataset = process_dataset(dataset, tokenizer)
    tokenizer.train(False)
    save(tokenizer, os.path.join(cfg['tokenizer_path'], cfg['data_name']))
    return


if __name__ == "__main__":
    main()
