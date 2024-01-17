from config import cfg
from .stats import make_stats


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['target_mode'] = cfg['control']['target_mode']
    cfg['model_name'] = cfg['control']['model_name']

    batch_size = {'ML100K': 100, 'ML1M': 500, 'ML10M': 1000, 'ML20M': 1000, 'Douban': 100, 'Amazon': 500}
    cfg['batch_size'] = batch_size[cfg['data_name']]
    cfg['step_period'] = 1
    cfg['num_steps'] = 80000
    cfg['eval_period'] = 200
    cfg['num_epochs'] = 2
    if cfg['model_name'] == 'base':
        cfg['num_epochs'] = 200
    cfg['collate_mode'] = 'dict'

    cfg['model'] = {}
    cfg['model']['target_mode'] = cfg['target_mode']
    cfg['model']['model_name'] = cfg['model_name']
    cfg['model']['base'] = {}
    cfg['model']['mf'] = {'hidden_size': 256}
    cfg['model']['nmf'] = {'hidden_size': [256, 128]}
    cfg['model']['ae'] = {'encoder_hidden_size': [256, 128], 'decoder_hidden_size': [128, 256]}
    cfg['model']['simplex'] = {'hidden_size': 256}
    cfg['model']['stats'] = make_stats()['{}_{}'.format(cfg['data_name'], cfg['target_mode'])]

    tag = cfg['tag']
    cfg[tag] = {}
    cfg[tag]['optimizer'] = {}
    cfg[tag]['optimizer']['optimizer_name'] = 'AdamW'
    cfg[tag]['optimizer']['lr'] = 3e-4
    cfg[tag]['optimizer']['momentum'] = 0.9
    cfg[tag]['optimizer']['betas'] = (0.9, 0.999)
    cfg[tag]['optimizer']['weight_decay'] = 5e-4
    cfg[tag]['optimizer']['nesterov'] = True
    cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
    cfg[tag]['optimizer']['step_period'] = cfg['step_period']
    cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
    cfg[tag]['optimizer']['scheduler_name'] = 'CosineAnnealingLR'
    return
