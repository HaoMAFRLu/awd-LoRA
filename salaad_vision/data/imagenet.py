"""Build a local-smoke or cluster ImageNet DataLoader from explicit config."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, Union

from loguru import logger
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

_VALID_LOCATIONS = frozenset({"local_smoke", "cluster_snapshot"})
_VALID_SPLITS = frozenset({"train", "validation", "test"})

VisionSample = Dict[str, Union[Tensor, int]]


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
    if isinstance(value, (str, Path)):
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

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)


def _build_dataloader(
    dataset: Any,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    streaming_dataset = _StreamingImageNetDataset(
        dataset,
        transform=_build_transform(split),
    )
    return DataLoader(
        streaming_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def build_imagenet_dataloader(
    config: Dict[str, Any],
    *,
    rank: int,
    world_size: int,
) -> DataLoader:
    """Build the local or cluster ImageNet DataLoader selected in config."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    data_config = config.get("data", {})
    dataset = data_config.get("dataset")

    if dataset != "ILSVRC/imagenet-1k":
        raise ValueError(f"unsupported vision dataset: {dataset!r}")

    location = data_config.get("location")
    if location not in _VALID_LOCATIONS:
        raise ValueError(f"unsupported ImageNet data location: {location!r}")

    root_value = data_config.get("root")
    if not root_value:
        raise ValueError("vision data requires data.root")
    root = Path(root_value).expanduser()

    batch_size = config.get("batch_size", 1)
    num_workers = config.get("num_workers", 0)
    split = data_config.get("split")
    if split not in _VALID_SPLITS:
        raise ValueError(
            f"split must be one of {sorted(_VALID_SPLITS)}, got {split!r}"
        )
    if data_config.get("streaming") is not True:
        raise ValueError("ImageNet parquet loading requires streaming=True")

    cache_dir_value = data_config.get("cache_dir")
    if not cache_dir_value:
        raise ValueError("ImageNet parquet loading requires data.cache_dir")
    cache_dir = Path(cache_dir_value).expanduser()

    data_directory = root / "data"
    if not data_directory.is_dir():
        raise FileNotFoundError(
            f"ImageNet snapshot data directory does not exist: {data_directory}"
        )
    shards = sorted(data_directory.glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"no {split!r} parquet shards found under {data_directory}"
        )
    rank_shards = shards[rank::world_size]
    if not rank_shards:
        raise ValueError(
            f"rank {rank} received no {split!r} shards: "
            f"{len(shards)} shards cannot cover world_size={world_size}"
        )
    logger.info(
        "[Rank {}] assigned {} of {} ImageNet {} shards",
        rank,
        len(rank_shards),
        len(shards),
        split,
    )

    from datasets import Image as HuggingFaceImage
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "parquet",
        data_files={split: [str(shard) for shard in rank_shards]},
        split=split,
        streaming=True,
        cache_dir=str(cache_dir),
    )
    dataset = dataset.cast_column(
        "image",
        HuggingFaceImage(decode=False),
    )
    logger.info(
        "[Rank {}] ImageNet loader uses encoded image bytes with explicit RGB "
        "decoding; Hugging Face EXIF/XMP auto-decoding is disabled",
        rank,
    )
    shuffle = data_config.get("shuffle", split == "train")
    if shuffle:
        shuffle_buffer_size = data_config.get("shuffle_buffer_size", 10_000)
        if shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be positive")
        dataset = dataset.shuffle(
            seed=config.get("seed_for_shuffle", config.get("seed", 0)),
            buffer_size=shuffle_buffer_size,
        )

    return _build_dataloader(
        dataset,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=data_config.get(
            "pin_memory",
            location == "cluster_snapshot",
        ),
        drop_last=data_config.get("drop_last", split == "train"),
    )
