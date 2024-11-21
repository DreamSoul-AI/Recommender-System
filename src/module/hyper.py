from config import cfg
from .stats import make_stats


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']
    cfg['score_mode'] = cfg['control']['score_mode']
    cfg['loss_mode'] = cfg['control']['loss_mode']

    batch_size = {'AmazonBeauty': 512}
    cfg['batch_size'] = batch_size[cfg['data_name']]
    cfg['step_period'] = 1
    cfg['num_steps'] = 1
    cfg['eval_period'] = 30
    cfg['num_epochs'] = 100
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
    cfg['model']['simplex'] = {'hidden_size': 64, 'aggregation_mode': 'mean', 'gamma': 1, 'attention_dropout': 0.,
                               'net_dropout': 0.}
    cfg['model']['gru4rec'] = {'hidden_size': 256}
    cfg['model']['youtubednn'] = {'hidden_size': 256}
    if 'make_stats' not in cfg:
        cfg['model']['stats'] = make_stats('{}'.format(cfg['data_name']))
    cfg['model']['num_negatives'] = 1000
    cfg['model']['pad_token'] = -100
    cfg['model']['enable_bias'] = False

    loss_kwargs = {'contrastive': {'margin': 0.9, 'negative_weight': 150}, 'margin': {'margin': 0.9}}
    cfg['model']['loss_kwargs'] = loss_kwargs.get(cfg['loss_mode'], None)

    tag = cfg['tag']
    cfg[tag] = {}
    cfg[tag]['optimizer'] = {}
    cfg[tag]['optimizer']['optimizer_name'] = 'AdamW'
    cfg[tag]['optimizer']['lr'] = 1e-4
    cfg[tag]['optimizer']['momentum'] = 0.9
    cfg[tag]['optimizer']['betas'] = (0.9, 0.999)
    cfg[tag]['optimizer']['weight_decay'] = 5e-4
    cfg[tag]['optimizer']['nesterov'] = True
    cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
    cfg[tag]['optimizer']['step_period'] = cfg['step_period']
    cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
    cfg[tag]['optimizer']['scheduler_name'] = 'CosineAnnealingLR'
    return
