from config import cfg
from .stats import make_stats


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']
    cfg['embedding_mode'] = cfg['control']['embedding_mode']
    cfg['loss_mode'] = cfg['control']['loss_mode']

    batch_size = {'AmazonBeauty': 256}
    cfg['batch_size'] = batch_size[cfg['data_name']]
    cfg['step_period'] = 1
    cfg['num_steps'] = 30
    cfg['eval_period'] = 30
    # cfg['num_epochs'] = 1
    cfg['collate_mode'] = 'dict'

    cfg['max_length_mode'] = 'longest'

    cfg['model'] = {}
    cfg['model']['model_name'] = cfg['model_name']
    cfg['model']['embedding_mode'] = cfg['embedding_mode']
    cfg['model']['loss_mode'] = cfg['loss_mode']
    cfg['model']['mf'] = {'hidden_size': 64}
    cfg['model']['simplex'] = {'hidden_size': 64, 'aggregation_mode': 'mean', 'gamma': 1, 'attention_dropout': 0.,
                               'net_dropout': 0.1, 'enable_bias': True}
    cfg['model']['new_simplex'] = {'hidden_size': 64, 'aggregation_mode': 'user_attention', 'gamma': 1, 'attention_dropout': 0.,
                               'net_dropout': 0.1, 'enable_bias': True}
    cfg['model']['gru4rec'] = {'hidden_size': [256, 128, 64]}
    cfg['model']['youtubednn'] = {'hidden_size': [256, 128, 64]}
    cfg['model']['dssm'] = {'hidden_size': [256, 128, 64]}
    cfg['model']['dssm_senet'] = {'hidden_size': [256, 128, 64]}
    cfg['model']['sasrec'] = {'hidden_size': 64, 'dropout_rate': 0., 'num_blocks': 2, 'num_heads': 1}
    cfg['model']['mind'] = {'hidden_size': 16, 'interest_num': 4}
    cfg['model']['comirec'] = {'hidden_size': 16, 'interest_num': 4}
    cfg['model']['sine'] = {'hidden_size': 16, 'num_concepts': 20, 'num_intention': 2, 'hidden_att_dim': 512}
    cfg['model']['stamp'] = {'hidden_size': 64}
    cfg['model']['narm'] = {'hidden_size': 64}
    if 'make_stats' not in cfg:
        cfg['model']['stats'] = make_stats('{}'.format(cfg['data_name']))
    cfg['model']['num_negatives'] = 800
    cfg['model']['pad_token'] = -100
    cfg['model']['padding_idx'] = 0

    cfg['model']['loss_kwargs'] = {}
    loss_hyperparam = {'contrastive': {'margin': 0.3, 'negative_weight': 10}, 'margin': {'margin': 0.3}}
    cfg['model']['loss_kwargs']['loss_hyperparam'] = loss_hyperparam.get(cfg['loss_mode'], None)

    tag = cfg['tag']
    cfg[tag] = {}
    cfg[tag]['optimizer'] = {}
    cfg[tag]['optimizer']['optimizer_name'] = 'AdamW'
    cfg[tag]['optimizer']['lr'] = 1e-3
    cfg[tag]['optimizer']['momentum'] = 0.9
    cfg[tag]['optimizer']['betas'] = (0.9, 0.999)
    cfg[tag]['optimizer']['weight_decay'] = 1e-6
    cfg[tag]['optimizer']['nesterov'] = True
    cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
    cfg[tag]['optimizer']['step_period'] = cfg['step_period']
    cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
    # cfg[tag]['optimizer']['scheduler_name'] = 'CosineAnnealingLR'
    cfg[tag]['optimizer']['scheduler_name'] = 'None'
    return
