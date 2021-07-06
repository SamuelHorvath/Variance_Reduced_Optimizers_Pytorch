from torch.utils.data import DataLoader

from .base_class import VROptimizer
from utils.utils import loader_kwargs
from utils.loss import Loss


class SGD(VROptimizer):
    def __init__(self, name, model, train_set, batch_size, lr, device, num_workers, weight_decay):
        super().__init__(name, model, train_set, batch_size, lr, device, num_workers, weight_decay)
        small_batch_size = int(name.split('_')[1])
        self.train_loader = DataLoader(
            train_set, batch_size=small_batch_size, shuffle=True, num_workers=num_workers, **loader_kwargs)

    def run_one_iter(self):
        self.model.train()
        data, label = next(iter(self.train_loader))
        batch_size = data.shape[0]
        data, label = data.to(self.device), label.to(self.device)

        self.optimiser.zero_grad(set_to_none=True)
        output = self.model(data)
        loss = Loss.compute_loss(output, label, self.model)
        loss.backward()
        self.optimiser.step()

        it_budget = batch_size
        return it_budget, loss, batch_size, output, label
