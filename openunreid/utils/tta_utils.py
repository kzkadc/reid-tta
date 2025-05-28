from typing import Any, Callable
from dataclasses import dataclass, field, InitVar

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ignite.metrics import Metric

import numpy as np
import pandas as pd

from openunreid.core.utils.compute_dist import build_dist
from openunreid.core.metrics.rank import evaluate_rank
from openunreid.data.datasets import build_dataset
from openunreid.data.transformers import build_test_transformer
from openunreid.models.losses.entropy import Entropy


@dataclass
class ReIDMetrics(Metric):
    gallery_features: Tensor
    gallery_pids: Tensor
    gallery_cids: Tensor
    dist_cfg: Any
    output_transform: InitVar[Callable[..., tuple[Tensor, Tensor, Tensor]]]
    cmc_topk: list[int] = field(default_factory=lambda: [1, 5, 10])

    def __post_init__(self,
                      output_transform: Callable[..., tuple[Tensor, Tensor, Tensor]]):
        super().__init__(output_transform)
        if self.dist_cfg.norm_feat:
            norms: Tensor = self.gallery_features.norm(dim=1, keepdim=True)
            self.gallery_features = (self.gallery_features / norms).detach()

    def reset(self):
        self._dists: list[np.ndarray] = []
        self._pids: list[np.ndarray] = []
        self._cids: list[np.ndarray] = []

    def update(self, output: tuple[Tensor, Tensor, Tensor]):
        features, pids, cids = output

        if self.dist_cfg.norm_feat:
            norms = features.norm(dim=1, keepdim=True)
            features /= norms

        dist_mat: np.ndarray = build_dist(self.dist_cfg, features, self.gallery_features)
        self._dists.append(dist_mat)

        self._pids.append(pids.cpu().numpy())
        self._cids.append(cids.cpu().numpy())

    def compute(self) -> tuple[list[float], float]:
        dist_mat = np.concatenate(self._dists)
        pids = np.concatenate(self._pids)
        cids = np.concatenate(self._cids)

        gallery_pids = self.gallery_pids.cpu().numpy()
        gallery_cids = self.gallery_cids.cpu().numpy()

        cmc: np.ndarray
        mean_ap: float
        cmc, mean_ap = evaluate_rank(dist_mat, pids, gallery_pids,
                                     cids, gallery_cids,
                                     max_rank=max(self.cmc_topk),
                                     cmc_topk=self.cmc_topk)
        return [cmc[k - 1] for k in self.cmc_topk], mean_ap


class CMCDataFrame(pd.DataFrame):
    def __repr__(self) -> str:
        return "(CMC scores)"


class BatchCMC(ReIDMetrics):
    def reset(self):
        super().reset()
        self._batch_cmcs: list[np.ndarray] = []

    def update(self, output: tuple[Tensor, Tensor, Tensor]):
        features, pids, cids = output

        if self.dist_cfg.norm_feat:
            norms = features.norm(dim=1, keepdim=True)
            features /= norms

        gallery_pids = self.gallery_pids.cpu().numpy()
        gallery_cids = self.gallery_cids.cpu().numpy()

        dist_mat: np.ndarray = build_dist(self.dist_cfg, features, self.gallery_features)
        cmc: np.ndarray
        cmc, _ = evaluate_rank(dist_mat, pids, gallery_pids,
                               cids, gallery_cids,
                               max_rank=max(self.cmc_topk),
                               cmc_topk=self.cmc_topk,
                               verbose=False)

        idx = [c - 1 for c in self.cmc_topk]
        self._batch_cmcs.append(cmc[idx])

    def compute(self) -> CMCDataFrame:
        cmc = np.stack(self._batch_cmcs)
        return CMCDataFrame({
            f"top_{k}": cmc[:, i]
            for i, k in enumerate(self.cmc_topk)
        })


class EntropyDataFrame(pd.DataFrame):
    def __repr__(self) -> str:
        return "(Re-ID entropy)"

@dataclass
class ReIDEntropy(Metric):
    gallery_features: InitVar[Tensor]
    dist_cfg: Any
    ent_cfg: InitVar[Any]
    output_transform: InitVar[Callable[..., tuple[Tensor, Tensor, Tensor]]]

    def __post_init__(self,
                      gallery_features: Tensor,
                      ent_cfg: Any,
                      output_transform: Callable[..., tuple[Tensor, Tensor, Tensor]]):
        super().__init__(output_transform)
        if self.dist_cfg.norm_feat:
            norms: Tensor = gallery_features.norm(dim=1, keepdim=True)
            gallery_features = (gallery_features / norms).detach()

        self.entropy_func = Entropy(gallery_features=gallery_features, **ent_cfg)

    def reset(self):
        self._ent_list = []

    def update(self, features: Tensor):
        if self.dist_cfg.norm_feat:
            norms = features.norm(dim=1, keepdim=True)
            features /= norms

        ent: Tensor = self.entropy_func({"feat": features})
        self._ent_list.append(float(ent.item()))

    def compute(self) -> EntropyDataFrame:
        return EntropyDataFrame({
            "entropy": self._ent_list
        })


@torch.no_grad()
def extract_features(net: nn.Module, dataset: Dataset, cfg
                     ) -> tuple[Tensor, Tensor, Tensor]:
    net.eval()

    features = []
    pids = []
    cids = []

    dl = DataLoader(dataset, cfg.TEST.LOADER.samples_per_gpu)

    batch: dict[str, Tensor]
    for batch in dl:
        x = batch["img"].cuda()
        output: dict[str, Tensor] = net(x)
        features.append(output["feat"].cpu())

        pids.append(batch["id"])
        cids.append(batch["cid"])

    features = torch.cat(features)
    pids = torch.cat(pids)
    cids = torch.cat(cids)

    return features, pids, cids


def get_test_dataset(cfg, name: str, split: str,
                     corruption: tuple[str, float] | None = None) -> Dataset:
    print(f"get_test_dataset: {name}-{split}, corruption={corruption}", flush=True)
    data_root: str = cfg.DATA_ROOT
    dataset = build_dataset(
        name, data_root, split, del_labels=False,
        transform=build_test_transformer(cfg, corruption)
    )
    return dataset
