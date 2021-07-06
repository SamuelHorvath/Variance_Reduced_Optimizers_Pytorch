import torch


class Loss:
    criterion = None
    nc_regularizer = False
    nc_regularizer_value = 1e-3

    def __init__(self, criterion, nc_regularizer=False, nc_regularizer_value=1e-3):
        Loss.criterion = criterion
        Loss.nc_regularizer = nc_regularizer
        Loss.nc_regularizer_value = nc_regularizer_value

    @classmethod
    def compute_loss(cls, output, label, model):
        loss = cls.criterion(output, label)
        if Loss.nc_regularizer:
            flatten_params = torch.cat([p.view(-1) for p in model.parameters()])
            penalty = (cls.nc_regularizer_value / 2) * torch.sum(flatten_params**2 / (flatten_params**2 + 1.))
            loss += penalty
        return loss
