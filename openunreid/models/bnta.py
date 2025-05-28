import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class PPHead(nn.Module):
    def __init__(self, feat_dim: int, h: int):
        super().__init__()

        self.pp_fc = nn.Linear(feat_dim, h)

        self.h = h

    def forward(self, z: Tensor) -> Tensor:
        # (B,C,H,W) -> (B,C,h,h',W)
        z_parts: list[Tensor] = z.split(math.ceil(z.shape[2] / self.h), dim=2)    # (B,C,h',W) * h
        z_parts_pooled = [
            zp.mean(dim=(2, 3))
            for zp in z_parts
        ]   # (B,C)*h

        z_pooled = torch.stack(z_parts_pooled, dim=1)   # (B,h,C)
        pos_pred = self.pp_fc(z_pooled) # (B,h,h)
        return pos_pred


class PNNMHead(nn.Module):
    def __init__(self, in_dim:int, out_dim: int,
                 h: int, num_classes: int):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, num_classes)

        self.h = h

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        # (B,C,H,W) -> (B,C,h,h',W)
        z_parts: list[Tensor] = z.split(math.ceil(z.shape[2] / self.h), dim=2)    # (B,C,h',W) * h
        z_parts_pooled = [
            zp.mean(dim=(2, 3))
            for zp in z_parts
        ]   # (B,C)*h

        z_pooled = torch.stack(z_parts_pooled, dim=1)   # (B,h,C)
        z = self.fc1(z_pooled)   # (B,h,out_dim)
        y_pred = self.fc2(z)    # (B,h,num_classes)
        return z, y_pred


class PPLoss(nn.Module):
    def __init__(self, pp_head: PPHead):
        super().__init__()

        self.pp_head = pp_head

    def forward(self, results: dict[str, Tensor], *args) -> Tensor:
        h = self.pp_head.h

        z = results["backbone_feat"]
        pos_pred: Tensor = self.pp_head(z)   # (B,h,h)
        pos_pred = pos_pred.reshape(-1, h)  # (B*h,h)

        # [0, ..., h-1, 0, ..., h-1, 0,...]
        y = torch.arange(self.pp_head.h).tile([z.shape[0]])  # (B*h)
        y = y.to(pos_pred.device).long()

        loss = F.cross_entropy(pos_pred, y)
        return loss


class PNNMTrainLoss(nn.Module):
    def __init__(self, pnnm_head: PNNMHead):
        super().__init__()

        self.pnnm_head = pnnm_head

    def forward(self, results: dict[str, Tensor], targets: Tensor) -> Tensor:
        h = self.pnnm_head.h
        z = results["backbone_feat"]

        y_pred: Tensor
        _, y_pred = self.pnnm_head(z)   # (B,h,num_classes)
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])  # (B*h,num_classes)

        y = targets.repeat_interleave(h).to(z.device)    # (B*h)

        loss = F.cross_entropy(y_pred, y)
        return loss


class PNNMTTALoss(nn.Module):
    def __init__(self, pnnm_head: PNNMHead,
                 margin: float, k: int):
        super().__init__()

        self.pnnm_head = pnnm_head
        self.margin = margin
        self.k = k

    def forward(self, results: dict[str, Tensor], *args) -> Tensor:
        z: Tensor
        z, _ = self.pnnm_head(results["backbone_feat"]) # (B,h,d)
        z = z.permute(1, 0, 2)  # (h,B,d)

        dist_mat = torch.cdist(z, z)    # (h,B,B)

        nn_dist_mat: Tensor
        pos_dist_topk: Tensor
        neg_dist_topk: Tensor
        nn_dist_mat, _ = dist_mat.min(dim=0)  # (B,B)

        neg_dist_topk, _ = nn_dist_mat.flatten().topk(self.k * 2, largest=True)     # (k)

        # avoid extracting diagonal element
        nn_dist_mat = nn_dist_mat + torch.eye(nn_dist_mat.shape[0]).to(z.device) * nn_dist_mat.max()
        pos_dist_topk, _ = nn_dist_mat.flatten().topk(self.k * 2, largest=False)    # (k)

        loss = F.relu(self.margin + pos_dist_topk - neg_dist_topk).sum() / 2
        return loss
