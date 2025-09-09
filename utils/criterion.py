from torch import nn

import torch
from torch.nn import functional as F

def log_cosh_loss(pred, target, reduction='mean'):
    if reduction == 'none':
        return torch.log(torch.cosh(pred - target))
    return torch.mean(torch.log(torch.cosh(pred - target)))

def get_criterion(criterion_name, reduction=None):
        criterion = None
        if (criterion_name == 'mean_squared_error'):
            criterion = F.mse_loss
        elif (criterion_name == 'categorical_cross_entropy'):
            criterion = nn.CrossEntropyLoss(reduction=reduction)
        elif (criterion_name == 'mean_absolute_error'):
            criterion = F.l1_loss
        elif (criterion_name == 'binary_cross_entropy'):
            criterion = nn.BCELoss(reduction=reduction)
        elif (criterion_name == 'cross_entropy_loss'):
            criterion = nn.CrossEntropyLoss(reduction=reduction)
        elif (criterion_name == 'log_cosh_loss'):
            criterion = log_cosh_loss
        else:
            raise KeyError(f"Selected criterion: {criterion_name} is NOT available!")

        return criterion