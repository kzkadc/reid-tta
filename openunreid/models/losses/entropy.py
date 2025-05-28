from dataclasses import dataclass

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import numpy as np


def topk_dist(query_features: Tensor,
              gallery_features: Tensor,
              k: int,
              metric: str,
              selection: str = "top-k",
              on_gpu: bool = True) -> Tensor:
    Nq, Ng = query_features.shape[0], gallery_features.shape[0]
    q_idx, g_idx = torch.meshgrid(torch.arange(
        Nq), torch.arange(Ng), indexing="ij")

    if not on_gpu:
        query_features = query_features.cpu()
        gallery_features = gallery_features.cpu()

    # (Nq,Ng)
    match metric:
        case "sqeuclidean":
            dists = (query_features[q_idx] -
                     gallery_features[g_idx]).square().sum(dim=2)
        case "prod":
            dists = -(query_features[q_idx] *
                      gallery_features[g_idx]).sum(dim=2)
        case "cos":
            qf_norm: Tensor = query_features / \
                query_features.norm(dim=1, keepdim=True)
            gf_norm: Tensor = gallery_features / \
                gallery_features.norm(dim=1, keepdim=True)
            dists = -(qf_norm[q_idx] * gf_norm[g_idx]).sum(dim=2)
        case _:
            raise ValueError(f"Invalid distance metric: {metric!r}")

    match selection:
        case "top-k":
            k_nn_dists = dists.topk(k, dim=1, largest=False)[0]    # (Nq,k)
        case "bottom-k":
            k_nn_dists = dists.topk(k, dim=1, largest=True)[0]    # (Nq,k)
        case "top-bottom":
            top_dists = dists.topk(
                k // 2, dim=1, largest=False)[0]   # (Nq, k//2)
            btm_dists = dists.topk(
                k // 2, dim=1, largest=True)[0]    # (Nq, k//2)
            k_nn_dists = torch.cat([top_dists, btm_dists], dim=1)   # (Nq, k)
        case "random":
            idx = np.stack([
                np.random.choice(Ng, size=k, replace=False)
                for _ in range(Nq)
            ])
            idx = torch.from_numpy(idx).long()  # (Nq,k)
            dim = torch.arange(Nq)
            k_nn_dists = dists[:, idx][dim, dim]
        case _:
            raise ValueError(f"Invalid selection: {selection!r}")

    return k_nn_dists.cuda()


@dataclass
class Entropy(nn.Module):
    k: int
    gallery_features: Tensor
    metric: str
    selection: str = "top-k"
    on_gpu: bool = True
    sample_gallery: int = -1

    def __post_init__(self):
        super().__init__()

    def forward(self, results: dict[str, Tensor]) -> Tensor:
        if 0 < self.sample_gallery < self.gallery_features.shape[0]:
            idx = np.random.choice(
                self.gallery_features.shape[0], size=self.sample_gallery, replace=False)
            idx = torch.from_numpy(idx).long()
            gallery_features = self.gallery_features[idx]
        else:
            gallery_features = self.gallery_features

        k_nn_dists = topk_dist(
            results["feat"], gallery_features, self.k, self.metric, self.selection, self.on_gpu)
        prob = F.softmax(-k_nn_dists, dim=1)
        mean_ent = (-prob * F.log_softmax(-k_nn_dists, dim=1)
                    ).sum(dim=1).mean()
        return mean_ent


@dataclass
class InfoMax(nn.Module):
    k: int
    gallery_features: Tensor
    metric: str

    def __post_init__(self):
        super().__init__()

    def forward(self, results: dict[str, Tensor]) -> Tensor:
        # (B,k)
        k_nn_dists = topk_dist(
            results["feat"], self.gallery_features, self.k, self.metric)
        prob = F.softmax(-k_nn_dists, dim=1)
        mean_ent = (-prob * F.log_softmax(-k_nn_dists, dim=1)
                    ).sum(dim=1).mean()

        mean_prob = prob.mean(dim=0)
        ent_mean = (-mean_prob * mean_prob.log()).sum()

        return mean_ent - ent_mean


@dataclass
class FeatureLogitEntropy(nn.Module):
    key: str

    def __post_init__(self):
        super().__init__()

    def forward(self, results: dict[str, Tensor]) -> Tensor:
        feat = results[self.key]  # (B,D)
        pseudo_prob = F.softmax(feat, dim=1)
        mean_ent = (-pseudo_prob * F.log_softmax(feat, dim=1)
                    ).sum(dim=1).mean()
        return mean_ent
