from typing import Any, Iterator
import argparse
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path
from dataclasses import dataclass, InitVar
import itertools
import json

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel, DistributedDataParallel
from ignite.engine import Engine, State

import numpy as np

from openunreid.apis import test_reid, set_random_seed
from openunreid.data import build_test_dataloader
from openunreid.models import build_model
from openunreid.models.bnta import PPHead, PPLoss, PNNMHead, PNNMTTALoss
from openunreid.utils.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from openunreid.utils.dist_utils import init_dist, synchronize
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.logger import Logger
from openunreid.utils.tta_utils import ReIDMetrics, BatchCMC, get_test_dataset, extract_features


@dataclass
class TTAEngine(Engine):
    net: nn.Module
    pp_head: InitVar[PPHead]
    pnnm_head: InitVar[PNNMHead]
    opt: torch.optim.Optimizer
    lam_pp: float
    lam_pnnm: float
    margin: InitVar[float]
    k: InitVar[float]
    train_mode: bool
    gallery_features: InitVar[Tensor]
    gallery_pids: InitVar[Tensor]
    gallery_cids: InitVar[Tensor]
    metric_cfg: InitVar[Any]

    def __post_init__(self,
                      pp_head: PPHead,
                      pnnm_head: PNNMHead,
                      margin: float,
                      k: float,
                      gallery_features: Tensor,
                      gallery_pids: Tensor,
                      gallery_cids: Tensor,
                      metric_cfg: Any):
        super().__init__(self.update)

        ot = lambda d: (d["feature"], d["pid"], d["cid"])
        ReIDMetrics(gallery_features, gallery_pids, gallery_cids, metric_cfg, ot) \
            .attach(self, "reid_scores")
        BatchCMC(gallery_features, gallery_pids, gallery_cids, metric_cfg, ot) \
            .attach(self, "batch_cmc")

        self.pp_loss_fn = PPLoss(pp_head)
        self.pnnm_loss_fn = PNNMTTALoss(pnnm_head, margin, k)

    def update(self, engine: Engine, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.train_mode:
            self.net.train()
        else:
            self.net.eval()
        self.net.zero_grad()

        x = batch["img"].cuda()
        outputs: dict[str, Tensor] = self.net(x)

        loss: Tensor = self.lam_pp * self.pp_loss_fn(outputs) \
                        + self.lam_pnnm * self.pnnm_loss_fn(outputs)

        loss.backward()
        self.opt.step() # pylint: disable=E1120

        return {
            "loss": loss,
            "feature": outputs["feat"],
            "pid": batch["id"],
            "cid": batch["cid"]
        }


def parge_config():
    parser = argparse.ArgumentParser(description="Tent adaptation")
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
    parser.add_argument("--seed", type=int, default=-1)
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
    seed = args.seed if args.seed > 0 else cfg.TRAIN.seed
    set_random_seed(seed, cfg.TRAIN.deterministic)
    synchronize()

    # init logging file
    logger = Logger(cfg.work_dir / "log.txt", debug=False)
    sys.stdout = logger
    print(f"==========\nArgs:{args}\n==========")
    log_config_to_file(cfg)

    # build model
    cur_model = create_model(cfg, dist)

    # load heads
    pp_head, pnnm_head = create_heads(cfg)

    # build optimizer
    optimizer = get_optimizer(cfg, cur_model)

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

            gallery, gallery_features_subset = prepare_gallery(cfg, dataset, cur_model, gallery_corruption)
            cfg.TRAIN.LOSS.gallery_features = gallery_features_subset.cuda()

        state = adapt_once(cfg, dataset, query_corruption, cur_model, pp_head, pnnm_head, optimizer, gallery) # type: ignore

        print("online metrics", state.metrics)
        state.metrics["batch_cmc"].to_csv(
            str(cfg.work_dir / f"{i}_{dataset}_batch_cmc.csv"), index=False)

        print("offline test")
        cmc, mAP = test_reid(
            cfg, cur_model,
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
            {"state_dict": cur_model.state_dict()},
            str(cfg.work_dir / f"{i}_{dataset}_adapted_model.pt")
        )

    with (cfg.work_dir / "offline_metrics.json").open("w") as f:
        json.dump(offline_results, f, indent=4, ensure_ascii=False)

    # print time
    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))


def create_model(cfg, dist) -> nn.Module:
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

    return model


def create_heads(cfg) -> tuple[nn.Module, nn.Module]:
    head_state_dicts = torch.load(cfg.MODEL.heads_pretrained)
    num_classes = head_state_dicts["pnnm_head"]["fc2.weight"].shape[0]

    pp_head = PPHead(**cfg.TRAIN.PP_HEAD).cuda()
    pnnm_head = PNNMHead(num_classes=num_classes, **cfg.TRAIN.PNNM_HEAD).cuda()

    pp_head.load_state_dict(head_state_dicts["pp_head"])
    pnnm_head.load_state_dict(head_state_dicts["pnnm_head"])
    print(f"load heads: {cfg.MODEL.heads_pretrained}", flush=True)

    return pp_head, pnnm_head


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
               cur_model: nn.Module, pp_head: PPHead, pnnm_head: PNNMHead,
               optimizer: Optimizer,
               gallery: tuple[Tensor, Tensor, Tensor]) -> State:
    gallery_features, gallery_pids, gallery_cids = gallery
    tta_engine = TTAEngine(cur_model, pp_head, pnnm_head, optimizer,
                           gallery_features=gallery_features,
                           gallery_pids=gallery_pids,
                           gallery_cids=gallery_cids,
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
    print(f"{dataset}/num_iterations: ", tta_engine.state.iteration, flush=True)

    return tta_engine.state


def get_optimizer(cfg, net: nn.Module) -> torch.optim.Optimizer:
    match cfg.TRAIN.OPTIM.param:
        case "bn":
            params = itertools.chain.from_iterable(
                l.parameters()
                for l in extract_bn_layers(net)
            )
        case "all":
            params = net.parameters()
        case _ as name:
            raise ValueError(f"Invalid param: {name!r}")

    opt_name = cfg.TRAIN.OPTIM.optim
    opt_conf = cfg.TRAIN.OPTIM.config
    opt = eval(f"torch.optim.{opt_name}")(params, **opt_conf)
    return opt


def extract_bn_layers(module: nn.Module) -> Iterator[_BatchNorm]:
    for m in module.children():
        if isinstance(m, _BatchNorm):
            yield m
        else:
            yield from extract_bn_layers(m)


if __name__ == "__main__":
    main()
