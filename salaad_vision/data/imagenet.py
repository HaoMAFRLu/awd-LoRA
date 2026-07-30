"""Explicit local-smoke and cluster ImageNet DataLoader configurations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_IMAGENET_ROOT = (
    REPOSITORY_ROOT / "data" / "salaad_vision" / "smoke" / "imagenet_val64"
)
DEFAULT_CLUSTER_IMAGENET_ROOT = Path(
    "/lustre/fast/fast/hma2/data/imagenet2012/hf_snapshot"
)
DEFAULT_CLUSTER_DATASETS_CACHE = Path(
    "/lustre/fast/fast/hma2/data/imagenet2012/hf_datasets_cache"
)
LOCAL_IMAGENET_ROOT_ENV = "SALAAD_VISION_LOCAL_IMAGENET_ROOT"
CLUSTER_IMAGENET_ROOT_ENV = "SALAAD_VISION_CLUSTER_IMAGENET_ROOT"
CLUSTER_DATASETS_CACHE_ENV = "SALAAD_VISION_CLUSTER_DATASETS_CACHE"
_LEGACY_LOCAL_ROOT_ENV = "IMAGENET_SMOKE_ROOT"
_VALID_SPLITS = frozenset({"train", "validation", "test"})

PathLikeValue = Union[str, os.PathLike]
VisionSample = Dict[str, Union[Tensor, int]]


class ImageNetDataLocation(str, Enum):
    """Storage layout, selected explicitly rather than inferred from the host."""

    LOCAL_SMOKE = "local_smoke"
    CLUSTER_SNAPSHOT = "cluster_snapshot"


@dataclass(frozen=True)
class ImageNetLoaderConfig:
    """Everything needed to build one unambiguous ImageNet DataLoader."""

    location: ImageNetDataLocation
    root: Path
    split: str
    batch_size: int
    num_workers: int
    shuffle: bool
    cache_dir: Optional[Path] = None
    pin_memory: bool = False
    drop_last: bool = False
    seed: int = 0
    shuffle_buffer_size: int = 10_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser())
        if self.cache_dir is not None:
            object.__setattr__(
                self,
                "cache_dir",
                Path(self.cache_dir).expanduser(),
            )
        if self.split not in _VALID_SPLITS:
            raise ValueError(
                f"split must be one of {sorted(_VALID_SPLITS)}, got {self.split!r}"
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be positive")
        if self.location is ImageNetDataLocation.LOCAL_SMOKE:
            if self.split != "validation":
                raise ValueError("local_smoke only represents the validation split")
            if self.shuffle:
                raise ValueError("local_smoke must remain deterministic (shuffle=False)")

    @classmethod
    def local_smoke(
        cls,
        *,
        root: Optional[PathLikeValue] = None,
        batch_size: int = 8,
        num_workers: int = 0,
    ) -> "ImageNetLoaderConfig":
        if root is None:
            root = (
                os.environ.get(LOCAL_IMAGENET_ROOT_ENV)
                or os.environ.get(_LEGACY_LOCAL_ROOT_ENV)
                or DEFAULT_LOCAL_IMAGENET_ROOT
            )
        return cls(
            location=ImageNetDataLocation.LOCAL_SMOKE,
            root=Path(root),
            split="validation",
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

    @classmethod
    def cluster_snapshot(
        cls,
        *,
        root: Optional[PathLikeValue] = None,
        split: str = "train",
        batch_size: int = 1,
        num_workers: int = 8,
        shuffle: Optional[bool] = None,
        cache_dir: Optional[PathLikeValue] = None,
        pin_memory: bool = True,
        drop_last: Optional[bool] = None,
        seed: int = 0,
        shuffle_buffer_size: int = 10_000,
    ) -> "ImageNetLoaderConfig":
        if root is None:
            root = os.environ.get(
                CLUSTER_IMAGENET_ROOT_ENV,
                str(DEFAULT_CLUSTER_IMAGENET_ROOT),
            )
        if cache_dir is None:
            cache_dir = os.environ.get(
                CLUSTER_DATASETS_CACHE_ENV,
                str(DEFAULT_CLUSTER_DATASETS_CACHE),
            )
        if shuffle is None:
            shuffle = split == "train"
        if drop_last is None:
            drop_last = split == "train"
        return cls(
            location=ImageNetDataLocation.CLUSTER_SNAPSHOT,
            root=Path(root),
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            cache_dir=Path(cache_dir),
            pin_memory=pin_memory,
            drop_last=drop_last,
            seed=seed,
            shuffle_buffer_size=shuffle_buffer_size,
        )


def _build_transform(split: str) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.2, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )


class _OriginalImageNetLabel:
    def __init__(self, class_names: Iterable[str]) -> None:
        labels = []
        for class_name in class_names:
            prefix, separator, label = class_name.partition("_")
            if prefix != "class" or separator != "_" or not label.isdigit():
                raise ValueError(
                    "local smoke class directories must use names like class_0535; "
                    f"got {class_name!r}"
                )
            labels.append(int(label))
        self.labels: Tuple[int, ...] = tuple(labels)

    def __call__(self, local_target: int) -> int:
        return self.labels[local_target]


class _DictionaryDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> VisionSample:
        image, label = self.dataset[index]
        return {"pixel_values": image, "labels": int(label)}


def _decode_huggingface_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        if image_bytes is not None:
            with Image.open(BytesIO(image_bytes)) as image:
                return image.convert("RGB")
        image_path = value.get("path")
        if image_path:
            with Image.open(image_path) as image:
                return image.convert("RGB")
    if isinstance(value, (str, os.PathLike)):
        with Image.open(value) as image:
            return image.convert("RGB")
    raise TypeError(f"Unsupported Hugging Face image value: {type(value).__name__}")


class _StreamingImageNetDataset(IterableDataset):
    def __init__(self, dataset: Any, transform: transforms.Compose) -> None:
        self.dataset = dataset
        self.transform = transform

    def __iter__(self) -> Iterator[VisionSample]:
        for example in self.dataset:
            image = _decode_huggingface_image(example["image"])
            label = example.get("label", -1)
            yield {
                "pixel_values": self.transform(image),
                "labels": -1 if label is None else int(label),
            }


def _build_local_dataloader(config: ImageNetLoaderConfig) -> DataLoader:
    if not config.root.is_dir():
        raise FileNotFoundError(f"local ImageNet smoke root does not exist: {config.root}")
    image_folder = datasets.ImageFolder(
        config.root,
        transform=_build_transform(config.split),
    )
    image_folder.target_transform = _OriginalImageNetLabel(image_folder.classes)
    dataset = _DictionaryDataset(image_folder)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        persistent_workers=config.num_workers > 0,
    )


def _build_cluster_dataloader(config: ImageNetLoaderConfig) -> DataLoader:
    data_directory = config.root / "data"
    if not data_directory.is_dir():
        raise FileNotFoundError(
            f"cluster ImageNet snapshot data directory does not exist: {data_directory}"
        )
    shards = sorted(data_directory.glob(f"{config.split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"no {config.split!r} parquet shards found under {data_directory}"
        )

    from datasets import load_dataset

    if config.cache_dir is None:
        raise ValueError("cluster_snapshot requires an explicit datasets cache_dir")
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "parquet",
        data_files={config.split: [str(shard) for shard in shards]},
        split=config.split,
        streaming=True,
        cache_dir=str(config.cache_dir),
    )
    if config.shuffle:
        dataset = dataset.shuffle(
            seed=config.seed,
            buffer_size=config.shuffle_buffer_size,
        )
    streaming_dataset = _StreamingImageNetDataset(
        dataset,
        transform=_build_transform(config.split),
    )
    return DataLoader(
        streaming_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        persistent_workers=config.num_workers > 0,
    )


def build_imagenet_dataloader(config: ImageNetLoaderConfig) -> DataLoader:
    """Build the explicitly selected local or cluster ImageNet loader."""
    if config.location is ImageNetDataLocation.LOCAL_SMOKE:
        return _build_local_dataloader(config)
    if config.location is ImageNetDataLocation.CLUSTER_SNAPSHOT:
        return _build_cluster_dataloader(config)
    raise ValueError(f"unsupported ImageNet data location: {config.location!r}")
