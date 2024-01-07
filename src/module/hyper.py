from config import cfg


def process_control():
    cfg['collate_mode'] = 'dict'
    cfg['data_name'] = cfg['control']['data_name']
    cfg['target_mode'] = cfg['control']['target_mode']
    cfg['model_name'] = cfg['control']['model_name']
    cfg['max_length'] = 128
    cfg['base'] = {}
    cfg['mf'] = {'hidden_size': 256}
    cfg['nmf'] = {'hidden_size': [256, 128]}
    cfg['ae'] = {'encoder_hidden_size': [256, 128], 'decoder_hidden_size': [128, 256]}
    cfg['simplex'] = {'hidden_size': 256}
    batch_size = {'ML100K': 100, 'ML1M': 500, 'ML10M': 1000, 'ML20M': 1000, 'Douban': 100, 'Amazon': 500}
    model_name = cfg['model_name']
    cfg[model_name]['shuffle'] = {'train': True, 'test': False}
    cfg[model_name]['optimizer_name'] = 'AdamW'
    cfg[model_name]['lr'] = 3e-4
    cfg[model_name]['momentum'] = 0.9
    cfg[model_name]['weight_decay'] = 5e-4
    cfg[model_name]['nesterov'] = True
    cfg[model_name]['betas'] = (0.9, 0.999)
    cfg[model_name]['scheduler_name'] = 'CosineAnnealingLR'
    cfg[model_name]['min_lr'] = 1e-5
    cfg[model_name]['batch_size'] = {'train': batch_size[cfg['data_name']], 'test': batch_size[cfg['data_name']]}
    cfg[model_name]['num_epochs'] = 200 if model_name != 'base' else 1
    return
