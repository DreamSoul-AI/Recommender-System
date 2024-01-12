import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model
from .rs import rs
from .tokenizer import Tokenizer
from transformers import get_linear_schedule_with_warmup
from config import cfg


def make_model(model_name, tokenizer):
    model_ = eval('model.{}(tokenizer)'.format(model_name))
    model_ = rs(model_)
    return model_


def make_tokenizer():
    tokenizer = Tokenizer()
    return tokenizer


def make_loss(output, target):
    if cfg['target_mode'] == 'explicit':
        loss = F.mse_loss(output, target)
    elif cfg['target_mode'] == 'implicit':
        loss = F.binary_cross_entropy_with_logits(output, target)
    else:
        raise ValueError('Not valid target mode')
    return loss


def init_param(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if m.bias is not None:
            m.bias.data.zero_()
    return m


def make_optimizer(parameters, tag):
    if cfg[tag]['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(parameters, lr=cfg[tag]['lr'], momentum=cfg[tag]['momentum'],
                              weight_decay=cfg[tag]['weight_decay'], nesterov=cfg[tag]['nesterov'])
    elif cfg[tag]['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(parameters, lr=cfg[tag]['lr'], betas=cfg[tag]['betas'],
                               weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'AdamW':
        optimizer = optim.AdamW(parameters, lr=cfg[tag]['lr'], betas=cfg[tag]['betas'],
                                weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'LBFGS':
        optimizer = optim.LBFGS(parameters, lr=cfg[tag]['lr'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer, tag):
    if cfg[tag]['scheduler_name'] == 'None':
        scheduler = NoOpScheduler(optimizer)
    elif cfg[tag]['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg[tag]['step_size'], gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg[tag]['milestones'],
                                                   gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg[tag]['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_steps'],
                                                         eta_min=0)
    elif cfg[tag]['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg[tag]['factor'],
                                                         patience=cfg[tag]['patience'], verbose=False,
                                                         threshold=cfg[tag]['threshold'], threshold_mode='rel',
                                                         min_lr=cfg[tag]['min_lr'])
    elif cfg[tag]['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg[tag]['lr'], max_lr=10 * cfg[tag]['lr'])
    elif cfg['scheduler_name'] == 'LinearAnnealingLR':
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(cfg['num_steps'] * cfg['warmup_ratio']),
                                                    num_training_steps=cfg['num_steps'])
    elif cfg['scheduler_name'] == 'ConstantLR':
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=cfg['factor'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


class NoOpScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(NoOpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
