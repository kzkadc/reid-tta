import argparse
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from openunreid.apis import set_random_seed
from openunreid.models import build_model
from openunreid.utils.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from openunreid.utils.dist_utils import init_dist, synchronize
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.logger import Logger
from openunreid.utils.tta_utils import get_test_dataset, extract_features


def parge_config():
    parser = argparse.ArgumentParser(description="Feature extraction")
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

    for i, (dataset_name, corruption) in enumerate(zip(cfg.TEST.GALLERY.dataset, cfg.TEST.GALLERY.corruption)):
        match corruption:
            case (c,s):
                fn = cfg.work_dir / f"{i:d}_gallery_{c}_s={s}.pt"
            case None:
                fn = cfg.work_dir / f"{i:d}_gallery_clean.pt"
            case _:
                raise ValueError(f"Invalid corruption: {corruption}")
            
        gallery_dataset = get_test_dataset(cfg, dataset_name, "gallery", corruption)
        gallery_features, gallery_pids, _ = extract_features(model, gallery_dataset, cfg)
        print(f"gallery_features.shape: {gallery_features.shape}")
        
        torch.save({"features": gallery_features, "pids": gallery_pids}, str(fn))

    for i,(dataset_name, corruption) in enumerate(zip(cfg.TEST.QUERY.dataset,cfg.TEST.QUERY.corruption)):
        match corruption:
            case (c,s):
                fn = cfg.work_dir / f"{i:d}_query_{c}_s={s}.pt"
            case None:
                fn =  cfg.work_dir / f"{i:d}_query_clean.pt"
            case _:
                raise ValueError(f"Invalid corruption: {corruption}")
            
        query_dataset = get_test_dataset(cfg, dataset_name, "query", corruption)
        query_features, query_pids, _ = extract_features(model, query_dataset, cfg)
        print(f"query_features.shape: {query_features.shape}")

        torch.save({"features": query_features, "pids": query_pids}, str(fn))

    # print time
    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))


if __name__ == "__main__":
    main()
