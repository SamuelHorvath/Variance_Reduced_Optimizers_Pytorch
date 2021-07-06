import torch
from torch import nn
from torch.nn import DataParallel

import optims
from utils import models
from .logger import Logger
from .utils import init_metrics_meter, log_epoch_info, create_metrics_dict
from .data.data_loaders import get_num_classes
from .loss import Loss


def get_training_elements(model_name, loss_name, dataset, gpu, nc_regularizer, nc_regularizer_value):
    # Define the model
    model, current_epoch = initialise_model(
        model_name, dataset, use_pretrained=False)

    model = model_to_device(model, gpu)

    criterion = get_criterion(loss_name)
    Loss(criterion, nc_regularizer, nc_regularizer_value)
    return model, current_epoch


def get_vr_method(method_name, model, train_set, batch_size, lr, device, num_workers, weight_decay):
    if method_name.startswith('sgd'):
        vr_method_class = optims.SGD
    elif method_name == 'gd':
        vr_method_class = optims.GD
    elif method_name.startswith('svrg') or method_name.startswith('scsg'):
        vr_method_class = optims.SVRG
    elif method_name.startswith('sarah') or method_name.startswith('q-sarah') or method_name.startswith('e-sarah'):
        vr_method_class = optims.SARAH
    else:
        raise ValueError(f'{method_name} is not valid VR method.')
    return vr_method_class(method_name, model, train_set, batch_size, lr, device, num_workers, weight_decay)


def initialise_model(model_name, dataset, use_pretrained=False):
    if model_name == 'log_reg':
        model_name += '_' + dataset

    model = getattr(models, model_name)(pretrained=use_pretrained,
                                        num_classes=get_num_classes(dataset))
    current_epoch = 0
    return model, current_epoch


def model_to_device(model, device):
    if type(device) == list:  # if to allocate more than one GPU
        model = model.to(device[0])
        model = DataParallel(model, device_ids=device)
    else:
        model.to(device)
    return model


def run_one_step(i, spend_budget, budget, vr_method, metrics_meter, print_freq=50):
    it_budget, loss, batch_size, output, label = vr_method.run_one_iter()
    update_metrics(metrics_meter, loss, batch_size, output, label)
    if i % print_freq == 0:
        log_epoch_info(Logger, spend_budget, budget, metrics_meter, train=True)
    spend_budget += it_budget
    return spend_budget


def evaluate_model(model, loader, device, epoch,
                   print_freq=50, metric_to_optim='top_1'):
    metrics_meter = init_metrics_meter(epoch)
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(loader):
            batch_size = data.shape[0]
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = Loss.compute_loss(output, label, model)
            update_metrics(metrics_meter, loss, batch_size, output, label)
            if i % print_freq == 0:
                log_epoch_info(Logger, i, len(loader), metrics_meter, train=False)

    # Metric for avg/single model(s)
    Logger.get().info(f'{metric_to_optim}: {metrics_meter[metric_to_optim].get_avg()}')
    return create_metrics_dict(metrics_meter, train=False)


def accuracy(output, label, topk=(1,)):
    """
    Extract the accuracy of the model.
    :param output: The output of the model
    :param label: The correct target label
    :param topk: Which accuracies to return (e.g. top1, top5)
    :return: The accuracies requested
    """
    # logistic regression
    batch_size = label.size(0)
    if len(output.shape) == 2 and output.shape[1] == 1:
        pred = (output >= 0.5).float()
        return [(100. * torch.sum(pred == label) / batch_size).item()]

    # Cross Entropy loss
    maxk = max(topk)
    if len(output.size()) == 1:
        _, pred = output.topk(maxk, 0, True, True)
    else:
        _, pred = output.topk(maxk, 1, True, True)
    if pred.size(0) != 1:
        pred = pred.t()

    if pred.size() == (1,):
        correct = pred.eq(label)
    else:
        correct = pred.eq(label.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100. / batch_size).item())
    return res


def update_metrics(metrics_meter, loss, batch_size, output, label):
    metrics_meter['loss'].update(loss.item(), batch_size)
    acc = accuracy(output, label, (1,))
    metrics_meter['top_1_acc'].update(acc[0], batch_size)


def get_criterion(loss_name):
    if loss_name == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif loss_name == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Loss name {loss_name} is not defined.")
    return criterion

