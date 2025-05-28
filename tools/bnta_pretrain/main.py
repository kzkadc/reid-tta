import argparse
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
from torch import nn

from openunreid.apis import BaseRunner, test_reid, set_random_seed
from openunreid.core.solvers import build_lr_scheduler, build_optimizer
from openunreid.data import build_test_dataloader, build_train_dataloader
from openunreid.models import build_model
from openunreid.models.losses import build_loss
from openunreid.models.bnta import PPHead, PPLoss, PNNMHead, PNNMTrainLoss
from openunreid.utils.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from openunreid.utils.dist_utils import init_dist, synchronize
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.logger import Logger


def parge_config():
    parser = argparse.ArgumentParser(
        description="Pretraining on source-domain datasets"
    )
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
    logger = Logger(cfg.work_dir / "log.txt")
    sys.stdout = logger
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)

    # build train loader
    train_loader, _ = build_train_dataloader(cfg)

    # build model
    model = build_model(cfg, train_loader.loader.dataset.num_pids)
    model.cuda()

    if dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.gpu],
            output_device=cfg.gpu,
            find_unused_parameters=True,
        )
    elif cfg.total_gpus > 1:
        model = torch.nn.DataParallel(model)

    pp_head = PPHead(**cfg.TRAIN.PP_HEAD).cuda()
    pnnm_head = PNNMHead(num_classes=train_loader.loader.dataset.num_pids,
                         **cfg.TRAIN.PNNM_HEAD).cuda()

    # build optimizer
    optimizer = build_optimizer([model], **cfg.TRAIN.OPTIM)

    # build lr_scheduler
    if cfg.TRAIN.SCHEDULER.lr_scheduler is not None:
        lr_scheduler = build_lr_scheduler(optimizer, **cfg.TRAIN.SCHEDULER)
    else:
        lr_scheduler = None

    # build loss functions
    criterions = build_loss(
        cfg.TRAIN.LOSS, num_classes=train_loader.loader.dataset.num_pids, cuda=True
    )

    criterions["pp_loss"] = PPLoss(pp_head)
    criterions["pnnm_loss"] = PNNMTrainLoss(pnnm_head)

    cfg.TRAIN.LOSS.losses.update({
        "pp_loss": cfg.TRAIN.lam_pos,
        "pnnm_loss": cfg.TRAIN.lam_mat
    })

    # build runner
    runner = BaseRunner(
        cfg, model, optimizer, criterions, train_loader, lr_scheduler=lr_scheduler
    )

    # resume
    if args.resume_from:
        runner.resume(args.resume_from)

    # start training
    runner.run()

    # load the best model
    runner.resume(cfg.work_dir / "model_best.pth")

    # final testing
    test_loaders, queries, galleries = build_test_dataloader(cfg)
    for i, (loader, query, gallery) in enumerate(zip(test_loaders, queries, galleries)):
        cmc, mAP = test_reid(
            cfg, model, loader, query, gallery, dataset_name=cfg.TEST.datasets[i]
        )

    torch.save({
        "pp_head": pp_head.state_dict(),
        "pnnm_head": pnnm_head.state_dict()
    }, str(cfg.work_dir / "bnta_heads.pth"))

    # print time
    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))




if __name__ == "__main__":
    main()
