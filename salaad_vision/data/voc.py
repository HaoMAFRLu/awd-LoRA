"""Pascal VOC 2012 semantic-segmentation dataset and DataLoader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

_VALID_SPLITS = frozenset({"train", "val", "trainval"})
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

VisionSample = Dict[str, Tensor]


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _probability(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


def _range(
    value: Any,
    name: str,
    *,
    maximum: Union[int, float, None] = None,
) -> Tuple[float, float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        )
    ):
        raise ValueError(f"{name} must contain two numbers")
    result = (float(value[0]), float(value[1]))
    if result[0] <= 0 or result[0] > result[1]:
        raise ValueError(f"{name} must satisfy 0 < minimum <= maximum")
    if maximum is not None and result[1] > maximum:
        raise ValueError(f"{name} values must not exceed {maximum}")
    return result


def resolve_voc2012_root(root: Union[str, Path]) -> Path:
    """Resolve a root pointing at VOC2012, VOCdevkit, or their parent."""
    path = Path(root).expanduser()
    candidates = [
        path,
        path / "VOC2012",
        path / "VOCdevkit" / "VOC2012",
    ]
    required = (
        Path("JPEGImages"),
        Path("SegmentationClass"),
        Path("ImageSets") / "Segmentation",
    )
    for candidate in candidates:
        if all((candidate / relative).is_dir() for relative in required):
            return candidate
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "could not find a VOC2012 directory with JPEGImages, "
        f"SegmentationClass, and ImageSets/Segmentation; checked: {rendered}"
    )


class _PairedTransform:
    def __init__(
        self,
        *,
        train: bool,
        image_size: int,
        resize_size: int,
        crop_scale: Tuple[float, float],
        crop_ratio: Tuple[float, float],
        horizontal_flip_probability: float,
    ) -> None:
        self.train = train
        self.image_size = image_size
        self.resize_size = resize_size
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.horizontal_flip_probability = horizontal_flip_probability
        self.normalize = transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)

    def __call__(self, image: Image.Image, mask: Image.Image) -> VisionSample:
        if image.size != mask.size:
            raise ValueError(
                f"VOC image and mask sizes differ: {image.size} != {mask.size}"
            )

        if self.train:
            top, left, height, width = transforms.RandomResizedCrop.get_params(
                image,
                scale=self.crop_scale,
                ratio=self.crop_ratio,
            )
            image = TF.resized_crop(
                image,
                top,
                left,
                height,
                width,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            mask = TF.resized_crop(
                mask,
                top,
                left,
                height,
                width,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
            )
            if torch.rand(()).item() < self.horizontal_flip_probability:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
        else:
            image = TF.resize(
                image,
                self.resize_size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            mask = TF.resize(
                mask,
                self.resize_size,
                interpolation=InterpolationMode.NEAREST,
            )
            image = TF.center_crop(image, [self.image_size, self.image_size])
            mask = TF.center_crop(mask, [self.image_size, self.image_size])

        pixels = self.normalize(TF.to_tensor(image))
        labels = TF.pil_to_tensor(mask).squeeze(0).to(torch.int64)
        return {"pixel_values": pixels, "labels": labels}


class VOCSegmentationDataset(Dataset[VisionSample]):
    """Read the standard on-disk VOCdevkit/VOC2012 segmentation layout."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        transform: _PairedTransform,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(
                f"VOC split must be one of {sorted(_VALID_SPLITS)}, got {split!r}"
            )
        self.root = resolve_voc2012_root(root)
        self.split = split
        self.transform = transform

        split_file = self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"VOC split file does not exist: {split_file}")
        identifiers = [
            line.strip()
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not identifiers:
            raise ValueError(f"VOC split contains no samples: {split_file}")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"VOC split contains duplicate sample IDs: {split_file}")

        self.samples: List[Tuple[Path, Path]] = []
        for identifier in identifiers:
            image_path = self.root / "JPEGImages" / f"{identifier}.jpg"
            mask_path = self.root / "SegmentationClass" / f"{identifier}.png"
            if not image_path.is_file():
                raise FileNotFoundError(f"VOC image does not exist: {image_path}")
            if not mask_path.is_file():
                raise FileNotFoundError(f"VOC mask does not exist: {mask_path}")
            self.samples.append((image_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> VisionSample:
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(mask_path) as mask_file:
            mask = mask_file.copy()
        return self.transform(image, mask)


def build_voc2012_dataloader(
    config: Dict[str, Any],
    *,
    rank: int,
    world_size: int,
) -> DataLoader:
    """Build an official Pascal VOC 2012 semantic-segmentation DataLoader."""
    if (
        not isinstance(world_size, int)
        or isinstance(world_size, bool)
        or world_size <= 0
    ):
        raise ValueError("world_size must be a positive integer")
    if (
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or rank < 0
        or rank >= world_size
    ):
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    data_config = config.get("data", {})
    root_value = data_config.get("root")
    if not isinstance(root_value, (str, Path)) or not str(root_value):
        raise ValueError("VOC 2012 data requires data.root")

    split = data_config.get("split")
    if split not in _VALID_SPLITS:
        raise ValueError(
            f"VOC split must be one of {sorted(_VALID_SPLITS)}, got {split!r}"
        )
    image_size = _positive_int(data_config.get("image_size", 224), "data.image_size")
    resize_size = _positive_int(
        data_config.get("resize_size", round(256 * image_size / 224)),
        "data.resize_size",
    )
    if split != "train" and resize_size < image_size:
        raise ValueError("data.resize_size must be at least data.image_size")
    crop_scale = _range(
        data_config.get("crop_scale", (0.5, 1.0)),
        "data.crop_scale",
        maximum=1.0,
    )
    crop_ratio = _range(
        data_config.get("crop_ratio", (3.0 / 4.0, 4.0 / 3.0)),
        "data.crop_ratio",
    )
    flip_probability = _probability(
        data_config.get("horizontal_flip_probability", 0.5),
        "data.horizontal_flip_probability",
    )

    dataset = VOCSegmentationDataset(
        root_value,
        split,
        _PairedTransform(
            train=split == "train",
            image_size=image_size,
            resize_size=resize_size,
            crop_scale=crop_scale,
            crop_ratio=crop_ratio,
            horizontal_flip_probability=flip_probability,
        ),
    )

    batch_size = _positive_int(config.get("batch_size", 1), "batch_size")
    num_workers = config.get("num_workers", 0)
    if (
        not isinstance(num_workers, int)
        or isinstance(num_workers, bool)
        or num_workers < 0
    ):
        raise ValueError("num_workers must be a non-negative integer")

    shuffle = data_config.get("shuffle", split == "train")
    pin_memory = data_config.get("pin_memory", False)
    drop_last = data_config.get("drop_last", False)
    persistent_workers = data_config.get("persistent_workers", num_workers > 0)
    for value, name in (
        (shuffle, "data.shuffle"),
        (pin_memory, "data.pin_memory"),
        (drop_last, "data.drop_last"),
        (persistent_workers, "data.persistent_workers"),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be true or false")
    if persistent_workers and num_workers == 0:
        raise ValueError("data.persistent_workers requires num_workers > 0")

    seed = config.get("seed_for_shuffle", config.get("seed", 0))
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed_for_shuffle must be an integer")
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    generator = torch.Generator()
    generator.manual_seed(seed + rank)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        generator=generator,
    )
