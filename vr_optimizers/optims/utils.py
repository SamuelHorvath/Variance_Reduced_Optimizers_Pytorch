import torch
from utils.loss import Loss


@torch.no_grad()
def set_vr_grad(model, model_snapshot, large_batch_grad):
    for i, (p, p_snap) in enumerate(zip(model.parameters(), model_snapshot.parameters())):
        if p.grad is not None:
            p.grad = p.grad - p_snap.grad + large_batch_grad[i]


def get_full_gradient(model, data_loader, device, stat=False):
    loss = 0.
    num_samples = 0
    output_all = None
    label_all = None
    zero_grad(model)
    for data, label in data_loader:
        batch_size = data.shape[0]
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss += Loss.compute_loss(output, label, model) * batch_size
        num_samples += batch_size
        if stat:
            output_all = output if output_all is None else torch.cat([output_all, output])
            label_all = label if label_all is None else torch.cat([label_all, label])
    loss /= num_samples
    loss.backward()
    grad = [p.grad for p in model.parameters()]
    zero_grad(model)
    if stat:
        return grad, loss, num_samples, output_all, label_all
    else:
        return grad


def zero_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


def get_grad_norm(grad):
    norm_squared = 0.
    for g in grad:
        if g is not None:
            norm_squared += torch.sum(g**2)
    return norm_squared.item()
