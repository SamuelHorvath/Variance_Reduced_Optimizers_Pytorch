import numpy as np
from torch.utils.data import DataLoader, Subset
from copy import deepcopy

from .base_class import VROptimizer
from .utils import zero_grad, get_full_gradient, set_vr_grad
from utils.utils import loader_kwargs
from utils.loss import Loss


class SVRG(VROptimizer):
    def __init__(self, name, model, train_set, batch_size, lr, device, num_workers, weight_decay):
        self.model_snap = deepcopy(model)
        super().__init__(name, model, train_set, batch_size, lr, device, num_workers, weight_decay)
        self.method_name = name.split('_')[0]
        self.n = len(self.train_set)
        if self.method_name == 'svrg':
            self.j = None
            if len(name.split('_')) == 1:
                big_bs = self.n
            else:
                big_bs = int(name.split('_')[1])
            self.big_bs = int(np.min([big_bs, self.n]))
        else:  # SCSG
            if len(name.split('_')) > 1:
                self.n = int(name.split('_')[1])
            self.j = 1
            self.big_bs = int(np.min([np.ceil(self.j**(3/2)), self.n]))
        self.small_bs = int(np.ceil(self.big_bs**(2/3)))
        self.train_loader = DataLoader(
            train_set, batch_size=self.small_bs, shuffle=True, num_workers=num_workers, **loader_kwargs)
        self.big_grad = None
        self.p = self.small_bs / self.big_bs

    def run_one_iter(self):
        self.model.train()
        self.model_snap.train()
        it_budget = 0

        data, label = next(iter(self.train_loader))
        batch_size = data.shape[0]
        data, label = data.to(self.device), label.to(self.device)

        # random version with theoretical constant step-size
        if (self.big_grad is None) or np.random.rand() < self.p:
            it_budget += self.big_bs
            snap_subset = Subset(
                self.train_set, np.random.choice(len(self.train_set), self.big_bs, replace=False))
            snap_loader = DataLoader(
                snap_subset, batch_size=self.batch_size, num_workers=self.num_workers, **loader_kwargs)
            self.big_grad = get_full_gradient(self.model, snap_loader, self.device)
            self.model_snap.load_state_dict(deepcopy(self.model.state_dict()))
            if self.method_name == 'scsg':
                self.j += 1
                self.big_bs = int(np.min([np.ceil(self.j**(3/2)), self.n]))
                self.small_bs = int(np.min([self.j, np.ceil(self.n**(2/3))]))
                self.train_loader.batch_sampler.batch_size = self.small_bs
                self.p = self.small_bs / self.big_bs

        # snap gradient
        zero_grad(self.model_snap)
        output = self.model_snap(data)
        loss = Loss.compute_loss(output, label, self.model_snap)
        loss.backward()

        # local small gradient
        self.optimiser.zero_grad(set_to_none=True)
        output = self.model(data)
        loss = Loss.compute_loss(output, label, self.model)
        loss.backward()

        # adjust grad and do grad step
        set_vr_grad(self.model, self.model_snap, self.big_grad)
        self.optimiser.step()

        it_budget += 2 * batch_size
        return it_budget, loss, batch_size, output, label
