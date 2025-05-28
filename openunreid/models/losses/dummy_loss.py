import torch
from torch import nn, Tensor


class DummyLoss(nn.Module):
    def forward(self, *args, **kwargs) -> Tensor:
        return torch.tensor(0, dtype=torch.float, requires_grad=True).cuda()
