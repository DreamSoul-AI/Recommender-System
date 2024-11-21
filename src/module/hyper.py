from config import cfg
from .stats import make_stats


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']
    cfg['score_mode'] = cfg['control']['score_mode']
    cfg['loss_mode'] = cfg['control']['loss_mode']

    batch_size = {'ML100K': 100, 'ML1M': 500, 'ML10M': 1000, 'ML20M': 1000, 'Douban': 100, 'AmazonBeauty': 1000}
    cfg['batch_size'] = batch_size[cfg['data_name']]
    cfg['step_period'] = 1
    cfg['num_steps'] = 1
    cfg['eval_period'] = 30
    # cfg['num_epochs'] = 20
    if cfg['model_name'] == 'base':
        cfg['num_epochs'] = 1
    cfg['collate_mode'] = 'dict'

    cfg['max_length_mode'] = 'longest'

    cfg['model'] = {}
    cfg['model']['model_name'] = cfg['model_name']
    cfg['model']['score_mode'] = cfg['score_mode']
    cfg['model']['loss_mode'] = cfg['loss_mode']
    cfg['model']['base'] = {}
    cfg['model']['mf'] = {'hidden_size': 256}
    cfg['model']['nmf'] = {'hidden_size': [256, 128]}
    cfg['model']['ae'] = {'encoder_hidden_size': [256, 128], 'decoder_hidden_size': [128, 256]}
    cfg['model']['simplex'] = {'hidden_size': 256, 'aggregation_mode': 'mean'}
    cfg['model']['gru4rec'] = {'hidden_size': 256}
    cfg['model']['youtubednn'] = {'hidden_size': 256}
    if 'make_stats' not in cfg:
        cfg['model']['stats'] = make_stats('{}'.format(cfg['data_name']))
    cfg['model']['num_negatives'] = 31
    cfg['model']['pad_token'] = -100

    tag = cfg['tag']
    cfg[tag] = {}
    cfg[tag]['optimizer'] = {}
    cfg[tag]['optimizer']['optimizer_name'] = 'SGD'
    cfg[tag]['optimizer']['lr'] = 1e-1
    cfg[tag]['optimizer']['momentum'] = 0.9
    cfg[tag]['optimizer']['betas'] = (0.9, 0.999)
    cfg[tag]['optimizer']['weight_decay'] = 5e-4
    cfg[tag]['optimizer']['nesterov'] = True
    cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
    cfg[tag]['optimizer']['step_period'] = cfg['step_period']
    cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
    cfg[tag]['optimizer']['scheduler_name'] = 'CosineAnnealingLR'
    return
