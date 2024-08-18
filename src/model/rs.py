import torch
import torch.nn as nn
import torch.nn.functional as F
import model


class RecommenderSystem(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, input):
        output = {}
        user, item, rating, item_hist = input['user'], input['item'], input['rating'], input['item_hist']
        output['rating'] = self.base(user, item, rating, item_hist)
        output['loss'] = model.make_loss(output['rating'], rating)
        return output


def rs(model, cfg):
    model = RecommenderSystem(model)
    return model
