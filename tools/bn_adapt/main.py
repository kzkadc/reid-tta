from typing import Any
import argparse
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path
from dataclasses import dataclass, InitVar
import json

import numpy as np

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel, DistributedDataParallel
from ignite.engine import Engine, State

from openunreid.apis import test_reid, set_random_seed
from openunreid.data import build_test_dataloader
from openunreid.data.datasets import build_dataset
from openunreid.data.transformers import build_test_transformer
from openunreid.models import build_model
from openunreid.models.losses import build_loss
from openunreid.utils.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from openunreid.utils.dist_utils import init_dist, synchronize
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.logger import Logger
from openunreid.utils.tta_utils import ReIDMetrics, BatchCMC, ReIDEntropy, get_test_dataset, extract_features


@dataclass
class BNAdaptEngine(Engine):
    net: nn.Module
    gallery_features: InitVar[Tensor]
    gallery_pids: InitVar[Tensor]
    gallery_cids: InitVar[Tensor]
    metric_cfg: InitVar[Any]
    entropy_cfg: InitVar[Any]
    train_mode: bool = True

    def __post_init__(self,
                      gallery_features: Tensor,
                      gallery_pids: Tensor,
                      gallery_cids: Tensor,
                      metric_cfg: Any,
                      entropy_cfg: Any):
        super().__init__(self.update)

        ot = lambda d: (d["feature"], d["pid"], d["cid"])
        ReIDMetrics(gallery_features, gallery_pids, gallery_cids, metric_cfg, ot) \
            .attach(self, "reid_scores")
        BatchCMC(gallery_features, gallery_pids, gallery_cids, metric_cfg, ot) \
            .attach(self, "batch_cmc")

        ent_ot = lambda d: d["feature"]
        ReIDEntropy(gallery_features.cuda(), metric_cfg, entropy_cfg, ent_ot).attach(self, "batch_entropy")

        self.train_mode = bool(self.train_mode)
        print(f"BNAdapt: train_mode={self.train_mode}")

    @torch.no_grad()
    def update(self, engine: Engine, batch: dict[str, Tensor]):
        if self.train_mode:
            self.net.train()
        else:
            self.net.eval()

        x = batch["img"].cuda()
        outputs = self.net(x)
        feature = outputs["feat"]

        return {
            "feature": feature,
            "pid": batch["id"],
            "cid": batch["cid"]
        }


def parge_config():
    parser = argparse.ArgumentParser(description="BN adaptation")
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--work-dir", help="the dir to save logs and models", default=""
    )
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tcp-port", type=str, default="5017")
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )
    args = parser.parse_args()

    cfg_from_yaml_file(args.config, cfg)

    cfg.launcher = args.launcher
    cfg.tcp_port = args.tcp_port
    if not args.work_dir:
        args.work_dir = Path(args.config).stem
    cfg.work_dir = cfg.LOGS_ROOT / args.work_dir
    mkdir_if_missing(cfg.work_dir)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    shutil.copy(args.config, cfg.work_dir / "config.yaml")

    return args, cfg


def main():
    start_time = time.monotonic()

    # init distributed training
    args, cfg = parge_config()
    dist = init_dist(cfg)
    set_random_seed(cfg.TRAIN.seed, cfg.TRAIN.deterministic)
    synchronize()

    # init logging file
    logger = Logger(cfg.work_dir / "log.txt", debug=False)
    sys.stdout = logger
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)

    # build model
    model = build_model(cfg, 0, init=cfg.MODEL.source_pretrained)
    model.cuda()
    if dist:
        ddp_cfg = {
            "device_ids": [cfg.gpu],
            "output_device": cfg.gpu,
            "find_unused_parameters": True,
        }
        model = DistributedDataParallel(model, **ddp_cfg)
    elif cfg.total_gpus > 1:
        model = DataParallel(model)

    cfg.TRAIN.gallery_sampling = -1

    offline_results = []
    test_loaders, test_queries, test_galleries = build_test_dataloader(cfg)

    for i, dataset in enumerate(cfg.TEST.datasets):
        print(dataset)
        if cfg.TEST.QUERY_CORRUPTION.types[i] is None:
            query_corruption = None
        else:
            query_corruption = (
                cfg.TEST.QUERY_CORRUPTION.types[i],
                cfg.TEST.QUERY_CORRUPTION.factors[i]
            )

        if i == 0 or cfg.TRAIN.update_gallery:
            print("update gallery features")
            if cfg.TEST.GALLERY_CORRUPTION.types[i] is None:
                gallery_corruption = None
            else:
                gallery_corruption = (
                    cfg.TEST.GALLERY_CORRUPTION.types[i],
                    cfg.TEST.GALLERY_CORRUPTION.factors[i]
                )

            gallery, _ = prepare_gallery(cfg, dataset, model, gallery_corruption)

        state = adapt_once(cfg, dataset, query_corruption, model, gallery)    # type: ignore

        print("online metrics", state.metrics)
        state.metrics["batch_cmc"].to_csv(
            str(cfg.work_dir / f"{i}_{dataset}_batch_cmc.csv"), index=False)
        state.metrics["batch_entropy"].to_csv(
            str(cfg.work_dir / f"{i}_{dataset}_batch_entropy.csv"), index=False)

        print("offline test")
        cmc, mAP = test_reid(
            cfg, model,
            test_loaders[i], test_queries[i], test_galleries[i],
            dataset_name=dataset
        )
        offline_results.append({
            "dataset": dataset,
            "query_corruption": [
                cfg.TEST.QUERY_CORRUPTION.types[i],
                cfg.TEST.QUERY_CORRUPTION.factors[i]
            ],
            "gallery_corruption": [
                cfg.TEST.GALLERY_CORRUPTION.types[i],
                cfg.TEST.GALLERY_CORRUPTION.factors[i]
            ],
            "CMC": cmc.tolist(),
            "mAP": float(mAP)
        })

        # save model
        torch.save(
            {"state_dict": model.state_dict()},
            str(cfg.work_dir / f"{i}_{dataset}_adapted_model.pt")
        )

    with (cfg.work_dir / "offline_metrics.json").open("w") as f:
        json.dump(offline_results, f, indent=4, ensure_ascii=False)

    # print time
    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))


def prepare_gallery(cfg, dataset: str, model: nn.Module,
                    corruption: tuple[str, float] | None) -> tuple[tuple[Tensor, Tensor, Tensor], Tensor]:
    print("extract gallery features", flush=True)
    gallery_dataset = get_test_dataset(cfg, dataset, "gallery", corruption)
    gallery_features, gallery_pids, gallery_cids = extract_features(model, gallery_dataset, cfg)

    samp = cfg.TRAIN.gallery_sampling
    if 0 < samp < gallery_features.shape[0]:
        print("sampling gallery")
        idx = np.random.choice(gallery_features.shape[0], size=samp, replace=False)
        idx = torch.from_numpy(idx).long()
        gallery_features_subset = gallery_features[idx].clone()
    else:
        gallery_features_subset = gallery_features

    print("sampled gallery feature shape:", gallery_features_subset.shape)

    return (gallery_features, gallery_pids, gallery_cids), gallery_features_subset


def adapt_once(cfg, dataset: str, corruption: tuple[str, float] | None,
               model: nn.Module, gallery: tuple[Tensor, Tensor, Tensor]) -> State:
    gallery_features, gallery_pids, gallery_cids = gallery
    tta_engine = BNAdaptEngine(model, gallery_features, gallery_pids, gallery_cids,
                               **cfg.TRAIN.ENGINE)

    query_loader = DataLoader(get_test_dataset(cfg, dataset, "query", corruption),
                              batch_size=cfg.TRAIN.LOADER.samples_per_gpu,
                              drop_last=True,
                              shuffle=True)

    # start adaptation
    print("start adaptation", flush=True)
    adapt_start_time = time.monotonic()
    tta_engine.run(query_loader)
    adapt_end_time = time.monotonic()
    adapt_time = adapt_end_time - adapt_start_time

    tta_engine.state.metrics["adapt_time"] = adapt_time

    print(f"{dataset}/adaptation_time: ", timedelta(seconds=adapt_time))
    print(f"{dataset}/num_iterations: ", tta_engine.state.iteration)

    return tta_engine.state


if __name__ == "__main__":
    main()
