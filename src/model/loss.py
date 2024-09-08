import torch.nn as nn
import torch
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0, negative_weight=None):
        """
        :param margin: float, margin in loss
        :param num_negs: int, number of negative samples
        :param negative_weight:, float, the weight set to the negative samples. When negative_weight=None, it
            equals to num_negs
        """
        super().__init__()
        self._margin = margin
        self._negative_weight = negative_weight

    def forward(self, y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        pos_logits = y_pred[:, 0]
        pos_loss = torch.relu(1 - pos_logits)
        neg_logits = y_pred[:, 1:]
        neg_loss = torch.relu(neg_logits - self._margin)
        if self._negative_weight:
            loss = pos_loss + neg_loss.mean(dim=-1) * self._negative_weight
        else:
            loss = pos_loss + neg_loss.mean(dim=-1)
        return loss.mean()


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        pos_logits = y_pred[:, 0]
        pos_loss = torch.pow(pos_logits - 1, 2) / 2
        neg_logits = y_pred[:, 1:]
        neg_loss = torch.pow(neg_logits, 2).mean(dim=-1) / 2
        loss = pos_loss + neg_loss
        return loss.mean()


class LogisticLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        pos_logits = y_pred[:, 0].unsqueeze(-1)
        neg_logits = y_pred[:, 1:]
        logits_diff = pos_logits - neg_logits
        loss = -torch.log(torch.sigmoid(logits_diff)).mean()
        return loss


class MarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        :param margin: float, margin in loss
        """
        super().__init__()
        self._margin = margin

    def forward(self, y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        pos_logits = y_pred[:, 0].unsqueeze(-1)
        neg_logits = y_pred[:, 1:]
        loss = torch.relu(self._margin + neg_logits - pos_logits).mean()
        return loss


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        logits = y_pred.flatten()
        labels = y_true.flatten().float()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        probs = F.softmax(y_pred, dim=1)
        hit_probs = probs[:, 0]
        loss = -torch.log(hit_probs).mean()
        return loss


def make_loss_fn(loss_mode):
    if loss_mode == 'contrastive':
        loss_fn = ContrastiveLoss()
    elif loss_mode == 'mse':
        loss_fn = MSELoss()
    elif loss_mode == 'logistic':
        loss_fn = LogisticLoss()
    elif loss_mode == 'margin':
        loss_fn = MarginLoss()
    elif loss_mode == 'bce':
        loss_fn = BinaryCrossEntropyLoss()
    elif loss_mode == 'ce':
        loss_fn = CrossEntropyLoss()
    else:
        raise ValueError(f'Loss mode {loss_mode} not supported')
    return loss_fn
