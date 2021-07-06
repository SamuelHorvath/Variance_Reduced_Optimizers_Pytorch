from torch.utils.data import DataLoader

from .base_class import VROptimizer
from .utils import get_full_gradient
from utils.utils import loader_kwargs


class GD(VROptimizer):
    def __init__(self, name, model, train_set, batch_size, lr, device, num_workers, weight_decay):
        super().__init__(name, model, train_set, batch_size, lr, device, num_workers, weight_decay)
        self.train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, **loader_kwargs)

    def run_one_iter(self):
        self.model.train()
        grad, loss, batch_size, output, label = get_full_gradient(
            self.model, self.train_loader, self.device, stat=True)
        for i, p in enumerate(self.model.parameters()):
            p.grad = grad[i]
        self.optimiser.step()
        it_budget = batch_size
        return it_budget, loss, batch_size, output, label
