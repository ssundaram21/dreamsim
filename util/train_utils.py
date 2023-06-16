import torch
from pytorch_lightning import seed_everything
import logging
from torchmetrics import Metric


class HingeLoss(torch.nn.Module):
    def __init__(self, device, margin):
        super(HingeLoss, self).__init__()
        self.device = device
        self.margin = margin

    def forward(self, x, y):
        y_rounded = torch.round(y) # Map [0, 1] -> {0, 1}
        y_transformed = -1 * (1 - 2 * y_rounded) # Map {0, 1} -> {-1, 1}
        return torch.max(torch.zeros(x.shape).to(self.device), self.margin + (-1 * (x * y_transformed))).sum()


def seed_worker(worker_id):
    clear_logging()
    worker_seed = torch.initial_seed() % 2 ** 32
    seed_everything(worker_seed)


def clear_logging():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if 'lightning' in name]
    for logger in loggers:
        logger.setLevel(logging.ERROR)


class Mean(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("vals", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denom", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, val, denom):
        self.vals = self.vals + val
        self.denom = self.denom + denom

    def compute(self):
        return self.vals / self.denom
