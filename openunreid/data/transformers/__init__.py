# Modified from https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/data/transforms/build.py  # noqa

from dataclasses import dataclass

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from PIL import Image
import numpy as np

from .auto_augment import ImageNetPolicy
from .gaussian_blur import GaussianBlur
from .mutual_transformer import MutualTransform
from .random_erasing import RandomErasing

__all__ = ["build_train_transformer", "build_test_transformer"]


def build_train_transformer(cfg):

    res = []

    # auto augmentation
    if cfg.DATA.TRAIN.is_autoaug:
        total_iters = cfg.TRAIN.epochs * cfg.TRAIN.iters
        res.append(ImageNetPolicy(total_iters))

    # resize
    res.append(T.Resize((cfg.DATA.height, cfg.DATA.width), interpolation=InterpolationMode.BICUBIC))

    # horizontal filp
    if cfg.DATA.TRAIN.is_flip:
        res.append(T.RandomHorizontalFlip(p=cfg.DATA.TRAIN.flip_prob))

    # padding
    if cfg.DATA.TRAIN.is_pad:
        res.extend(
            [
                T.Pad(cfg.DATA.TRAIN.pad_size),
                T.RandomCrop((cfg.DATA.height, cfg.DATA.width)),
            ]
        )

    # gaussian blur
    if cfg.DATA.TRAIN.is_blur:
        res.append(
            T.RandomApply([GaussianBlur([0.1, 2.0])], p=cfg.DATA.TRAIN.blur_prob)
        )

    # totensor
    res.append(T.ToTensor())

    # normalize
    res.append(T.Normalize(mean=cfg.DATA.norm_mean, std=cfg.DATA.norm_std))

    # random erasing
    if cfg.DATA.TRAIN.is_erase:
        res.append(
            RandomErasing(
                probability=cfg.DATA.TRAIN.erase_prob, mean=cfg.DATA.norm_mean
            )
        )

    # mutual transform (for MMT)
    if cfg.DATA.TRAIN.is_mutual_transform:
        return MutualTransform(T.Compose(res), cfg.DATA.TRAIN.mutual_times)

    return T.Compose(res)


def build_test_transformer(cfg, corruption: tuple[str, float] | None = None):
    res = []

    # resize
    res.append(T.Resize((cfg.DATA.height, cfg.DATA.width), interpolation=InterpolationMode.BICUBIC))

    # corrupt
    match corruption:
        case ("brightness", s):
            res.append(AdjustBrightness(s))
        case ("gaussian_blur", s):
            res.append(GaussianBlur(int(s)))
        case ("gaussian_noise", s):
            res.append(GaussianNoise(s))
        case ("pixelate", s):
            res.append(Pixelate(s))
        case (c, _):
            raise ValueError(f"Invalid corruption: {c!r}")

    # totensor
    res.append(T.ToTensor())

    # normalize
    res.append(T.Normalize(mean=cfg.DATA.norm_mean, std=cfg.DATA.norm_std))

    return T.Compose(res)


@dataclass
class AdjustBrightness:
    factor: float

    def __call__(self, img: Image.Image) -> Image.Image:
        img = T.functional.adjust_brightness(img, self.factor)
        return img


@dataclass
class GaussianBlur:
    k: int

    def __call__(self, img: Image.Image) -> Image.Image:
        img = T.functional.gaussian_blur(img, self.k)
        return img


@dataclass
class GaussianNoise:
    sigma: float

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.asarray(img).astype(float) / 255
        noise = np.random.normal(0.0, self.sigma, size=img_np.shape)
        img_np = np.clip(img_np + noise, 0.0, 1.0) * 255
        img = Image.fromarray(img_np.astype(np.uint8))
        return img


@dataclass
class Pixelate:
    factor: float

    def __call__(self, img: Image.Image) -> Image.Image:
        orig_h, orig_w = img.height, img.width
        h = int(orig_h * self.factor)
        w = int(orig_w * self.factor)
        img = img.resize((w, h), Image.BOX) \
                 .resize((orig_w, orig_h), Image.BOX)
        return img
