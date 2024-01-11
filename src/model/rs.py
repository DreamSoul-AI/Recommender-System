import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class RecommenderSystem(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input):
        output = {}
        return output


def rs(model, tokenizer):
    model = RecommenderSystem(model, tokenizer)
    return model
