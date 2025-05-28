# Written by Yixiao Ge

import torch.nn as nn

from .classification import CrossEntropyLoss, SoftEntropyLoss
from .memory import HybridMemory
from .triplet import SoftmaxTripletLoss, SoftSoftmaxTripletLoss, TripletLoss
from .gan_loss import GANLoss
from .sia_loss import SiaLoss
from .dummy_loss import DummyLoss
from .entropy import Entropy, InfoMax, FeatureLogitEntropy


def build_loss(
    cfg,
    num_classes=None,
    num_features=None,
    num_memory=None,
    triplet_key="pooling",
    cuda=False,
):

    criterions = {}
    for loss_name in cfg.losses.keys():
        match loss_name:
            case "cross_entropy":
                assert num_classes is not None
                criterion = CrossEntropyLoss(num_classes)

            case "soft_entropy":
                criterion = SoftEntropyLoss()

            case "triplet":
                if "margin" not in cfg:
                    cfg.margin = 0.3
                criterion = TripletLoss(
                    margin=cfg.margin, triplet_key=triplet_key
                )

            case "softmax_triplet":
                if "margin" not in cfg:
                    cfg.margin = 0.0
                criterion = SoftmaxTripletLoss(
                    margin=cfg.margin, triplet_key=triplet_key
                )

            case "soft_softmax_triplet":
                criterion = SoftSoftmaxTripletLoss(
                    triplet_key=triplet_key
                )

            case "hybrid_memory":
                assert num_features is not None and num_memory is not None
                if "temp" not in cfg:
                    cfg.temp = 0.05
                if "momentum" not in cfg:
                    cfg.momentum = 0.2
                criterion = HybridMemory(
                    num_features, num_memory, temp=cfg.temp, momentum=cfg.momentum
                )

            case 'recon':
                criterion = nn.L1Loss()

            case 'ide':
                criterion = nn.L1Loss()

            case 'dummy':
                criterion = DummyLoss()

            case 'entropy':
                if "selection" not in cfg:
                    print("'selection' of entropy loss is automatically set to 'top-k'")
                    cfg.selection = "top-k"
                criterion = Entropy(cfg.k, cfg.gallery_features, cfg.metric, cfg.selection, cfg.get("on_gpu", True))

            case 'feature_logit_entropy':
                criterion = FeatureLogitEntropy(cfg.key)

            case 'infomax':
                criterion = InfoMax(cfg.k, cfg.gallery_features, cfg.metric)

            case n if n.startswith('gan'):
                criterion = GANLoss('lsgan')

            case n if n.startswith('sia'):
                criterion = SiaLoss(margin=2.0)

            case _:
                raise KeyError("Unknown loss:", loss_name)

        criterions[loss_name] = criterion

    if cuda:
        for key in criterions.keys():
            criterions[key].cuda()

    return criterions
