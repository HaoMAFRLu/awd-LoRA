"""Tests for explicit local and cluster ImageNet loader configuration."""

from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader

from salad.register import get_data
from salaad_vision.data.imagenet import build_imagenet_dataloader

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SMOKE_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "salaad_vision"
    / "smoke"
    / "imagenet_val64_parquet"
)


def _write_validation_snapshot(
    root: Path,
    labels: list[int],
    *,
    shard_count: int = 1,
) -> None:
    if shard_count <= 0 or shard_count > len(labels):
        raise ValueError("shard_count must be between 1 and len(labels)")
    image_type = pa.struct(
        [
            pa.field("bytes", pa.binary()),
            pa.field("path", pa.string()),
        ]
    )
    data_directory = root / "data"
    data_directory.mkdir()
    for shard_index in range(shard_count):
        shard_labels = labels[shard_index::shard_count]
        images = []
        for label in shard_labels:
            image_buffer = BytesIO()
            Image.new("RGB", (256, 256), color=(label, 40, 60)).save(
                image_buffer,
                format="JPEG",
            )
            images.append(
                {
                    "bytes": image_buffer.getvalue(),
                    "path": f"sample_{label:04d}.jpg",
                }
            )
        pq.write_table(
            pa.table(
                {
                    "image": pa.array(images, type=image_type),
                    "label": pa.array(shard_labels, type=pa.int64()),
                }
            ),
            data_directory
            / f"validation-{shard_index:05d}-of-{shard_count:05d}.parquet",
        )


def _jpeg_with_invalid_xmp() -> bytes:
    image_buffer = BytesIO()
    Image.new("RGB", (256, 256), color=(20, 40, 60)).save(
        image_buffer,
        format="JPEG",
    )
    jpeg = image_buffer.getvalue()
    xmp_header = b"http://ns.adobe.com/xap/1.0/\x00"
    xmp_payload = (
        b'<?xpacket begin=""?>'
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"/>'
        b'<?xpacket end="w"?>'
        b"\xa8"
    )
    app1_payload = xmp_header + xmp_payload
    app1 = (
        b"\xff\xe1"
        + (len(app1_payload) + 2).to_bytes(2, "big")
        + app1_payload
    )
    return jpeg[:2] + app1 + jpeg[2:]


class ImageNetDataLoaderTest(unittest.TestCase):
    def test_get_data_preserves_text_dataset_path(self) -> None:
        dataset = Mock()
        shuffled_dataset = object()
        dataset.shuffle.return_value = shuffled_dataset
        config = {"seed_for_shuffle": 7}

        with patch("salad.register.datasets.load_dataset", return_value=dataset) as load:
            result = get_data(config, rank=0, world_size=1)

        load.assert_called_once_with(
            "allenai/c4",
            "en",
            split="train",
            streaming=True,
        )
        dataset.shuffle.assert_called_once_with(seed=7)
        self.assertIs(result, shuffled_dataset)

    def test_get_data_dispatches_local_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            _write_validation_snapshot(root, [17])
            config = {
                "batch_size": 1,
                "num_workers": 0,
                "data": {
                    "type": "vision",
                    "dataset": "ILSVRC/imagenet-1k",
                    "location": "local_smoke",
                    "root": temporary_root,
                    "cache_dir": str(root / "cache"),
                    "split": "validation",
                    "streaming": True,
                    "shuffle": False,
                },
            }
            loader = get_data(config, rank=0, world_size=1)
            batch = next(iter(loader))

        self.assertIsInstance(loader, DataLoader)
        self.assertEqual(tuple(batch["pixel_values"].shape), (1, 3, 224, 224))
        self.assertEqual(batch["labels"].tolist(), [17])

    def test_local_smoke_is_split_across_distributed_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            _write_validation_snapshot(
                root,
                list(range(4)),
                shard_count=4,
            )

            config = {
                "batch_size": 1,
                "num_workers": 0,
                "data": {
                    "type": "vision",
                    "dataset": "ILSVRC/imagenet-1k",
                    "location": "local_smoke",
                    "root": temporary_root,
                    "cache_dir": str(root / "cache"),
                    "split": "validation",
                    "streaming": True,
                    "shuffle": False,
                },
            }

            labels_by_rank = []
            for rank in range(2):
                loader = get_data(
                    config,
                    rank=rank,
                    world_size=2,
                )
                labels_by_rank.append(
                    {int(batch["labels"].item()) for batch in loader}
                )

        self.assertTrue(labels_by_rank[0].isdisjoint(labels_by_rank[1]))
        self.assertEqual(labels_by_rank[0] | labels_by_rank[1], set(range(4)))

    def test_single_shard_is_split_across_distributed_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            _write_validation_snapshot(root, list(range(6)))
            config = {
                "batch_size": 1,
                "num_workers": 0,
                "data": {
                    "type": "vision",
                    "dataset": "ILSVRC/imagenet-1k",
                    "location": "local_smoke",
                    "root": temporary_root,
                    "cache_dir": str(root / "cache"),
                    "split": "validation",
                    "streaming": True,
                    "shuffle": False,
                },
            }

            labels_by_rank = []
            for rank in range(2):
                loader = build_imagenet_dataloader(
                    config,
                    rank=rank,
                    world_size=2,
                )
                labels_by_rank.append(
                    {int(batch["labels"].item()) for batch in loader}
                )

        self.assertTrue(labels_by_rank[0].isdisjoint(labels_by_rank[1]))
        self.assertEqual(labels_by_rank[0] | labels_by_rank[1], set(range(6)))
        self.assertEqual([len(labels) for labels in labels_by_rank], [3, 3])

    def test_data_root_is_required(self) -> None:
        config = {
            "data": {
                "dataset": "ILSVRC/imagenet-1k",
                "location": "local_smoke",
                "split": "validation",
                "shuffle": False,
            }
        }
        with self.assertRaisesRegex(ValueError, "data.root"):
            build_imagenet_dataloader(config, rank=0, world_size=1)

    def test_cluster_snapshot_streaming_batch_contract(self) -> None:
        image_buffer = BytesIO()
        Image.new("RGB", (256, 256), color=(20, 40, 60)).save(
            image_buffer,
            format="JPEG",
        )
        image_type = pa.struct(
            [
                pa.field("bytes", pa.binary()),
                pa.field("path", pa.string()),
            ]
        )
        table = pa.table(
            {
                "image": pa.array(
                    [{"bytes": image_buffer.getvalue(), "path": None}],
                    type=image_type,
                ),
                "label": pa.array([17], type=pa.int64()),
            }
        )

        with tempfile.TemporaryDirectory() as temporary_root:
            data_directory = Path(temporary_root) / "data"
            data_directory.mkdir()
            pq.write_table(
                table,
                data_directory / "validation-00000-of-00001.parquet",
            )
            config = {
                "batch_size": 1,
                "num_workers": 0,
                "data": {
                    "dataset": "ILSVRC/imagenet-1k",
                    "location": "cluster_snapshot",
                    "root": temporary_root,
                    "cache_dir": str(Path(temporary_root) / "cache"),
                    "split": "validation",
                    "streaming": True,
                    "shuffle": False,
                    "pin_memory": False,
                },
            }
            loader = build_imagenet_dataloader(
                config,
                rank=0,
                world_size=1,
            )
            batch = next(iter(loader))

        self.assertEqual(tuple(batch["pixel_values"].shape), (1, 3, 224, 224))
        self.assertEqual(batch["labels"].tolist(), [17])

    def test_invalid_xmp_does_not_block_rgb_decode(self) -> None:
        image_bytes = _jpeg_with_invalid_xmp()
        with Image.open(BytesIO(image_bytes)) as image:
            with self.assertRaises(UnicodeDecodeError):
                image.getexif()

        image_type = pa.struct(
            [
                pa.field("bytes", pa.binary()),
                pa.field("path", pa.string()),
            ]
        )
        table = pa.table(
            {
                "image": pa.array(
                    [
                        {
                            "bytes": image_bytes,
                            "path": "invalid_xmp.JPEG",
                        }
                    ],
                    type=image_type,
                ),
                "label": pa.array([887], type=pa.int64()),
            }
        )

        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            data_directory = root / "data"
            data_directory.mkdir()
            pq.write_table(
                table,
                data_directory / "validation-00000-of-00001.parquet",
            )
            config = {
                "batch_size": 1,
                "num_workers": 0,
                "data": {
                    "dataset": "ILSVRC/imagenet-1k",
                    "location": "cluster_snapshot",
                    "root": temporary_root,
                    "cache_dir": str(root / "cache"),
                    "split": "validation",
                    "streaming": True,
                    "shuffle": False,
                    "pin_memory": False,
                },
            }
            loader = build_imagenet_dataloader(
                config,
                rank=0,
                world_size=1,
            )
            batch = next(iter(loader))

        self.assertEqual(tuple(batch["pixel_values"].shape), (1, 3, 224, 224))
        self.assertEqual(batch["labels"].tolist(), [887])

    def test_non_divisible_shards_are_assigned_once_across_ranks(self) -> None:
        image_buffer = BytesIO()
        Image.new("RGB", (256, 256), color=(20, 40, 60)).save(
            image_buffer,
            format="JPEG",
        )
        image_type = pa.struct(
            [
                pa.field("bytes", pa.binary()),
                pa.field("path", pa.string()),
            ]
        )

        with tempfile.TemporaryDirectory() as temporary_root:
            data_directory = Path(temporary_root) / "data"
            data_directory.mkdir()
            for label in range(5):
                table = pa.table(
                    {
                        "image": pa.array(
                            [{"bytes": image_buffer.getvalue(), "path": None}],
                            type=image_type,
                        ),
                        "label": pa.array([label], type=pa.int64()),
                    }
                )
                pq.write_table(
                    table,
                    data_directory
                    / f"validation-{label:05d}-of-00005.parquet",
                )

            config = {
                "seed_for_shuffle": 42,
                "batch_size": 1,
                "num_workers": 2,
                "data": {
                    "type": "vision",
                    "dataset": "ILSVRC/imagenet-1k",
                    "location": "cluster_snapshot",
                    "root": temporary_root,
                    "cache_dir": str(Path(temporary_root) / "cache"),
                    "split": "validation",
                    "streaming": True,
                    "shuffle": True,
                    "shuffle_buffer_size": 4,
                    "pin_memory": False,
                },
            }

            labels_by_rank = []
            for rank in range(2):
                loader = get_data(
                    config,
                    rank=rank,
                    world_size=2,
                )
                labels_by_rank.append(
                    {int(batch["labels"].item()) for batch in loader}
                )

        self.assertTrue(labels_by_rank[0].isdisjoint(labels_by_rank[1]))
        self.assertEqual(labels_by_rank[0] | labels_by_rank[1], set(range(5)))
        self.assertEqual([len(labels) for labels in labels_by_rank], [3, 2])

    def test_get_data_dispatches_cluster_snapshot(self) -> None:
        image_buffer = BytesIO()
        Image.new("RGB", (256, 256), color=(20, 40, 60)).save(
            image_buffer,
            format="JPEG",
        )
        image_type = pa.struct(
            [
                pa.field("bytes", pa.binary()),
                pa.field("path", pa.string()),
            ]
        )
        table = pa.table(
            {
                "image": pa.array(
                    [{"bytes": image_buffer.getvalue(), "path": None}],
                    type=image_type,
                ),
                "label": pa.array([23], type=pa.int64()),
            }
        )

        with tempfile.TemporaryDirectory() as temporary_root:
            data_directory = Path(temporary_root) / "data"
            data_directory.mkdir()
            pq.write_table(
                table,
                data_directory / "validation-00000-of-00001.parquet",
            )
            config = {
                "seed_for_shuffle": 42,
                "batch_size": 1,
                "num_workers": 0,
                "data": {
                    "type": "vision",
                    "dataset": "ILSVRC/imagenet-1k",
                    "location": "cluster_snapshot",
                    "root": temporary_root,
                    "cache_dir": str(Path(temporary_root) / "cache"),
                    "split": "validation",
                    "streaming": True,
                    "shuffle": False,
                    "pin_memory": False,
                },
            }
            loader = get_data(config, rank=0, world_size=1)
            batch = next(iter(loader))

        self.assertIsInstance(loader, DataLoader)
        self.assertEqual(tuple(batch["pixel_values"].shape), (1, 3, 224, 224))
        self.assertEqual(batch["labels"].tolist(), [23])

    def test_local_smoke_batch_contract(self) -> None:
        if not LOCAL_SMOKE_ROOT.is_dir():
            self.skipTest(f"local ImageNet smoke set is absent: {LOCAL_SMOKE_ROOT}")

        task_config = {
            "batch_size": 4,
            "num_workers": 0,
            "data": {
                "dataset": "ILSVRC/imagenet-1k",
                "location": "local_smoke",
                "root": str(LOCAL_SMOKE_ROOT),
                "cache_dir": str(LOCAL_SMOKE_ROOT / "cache"),
                "split": "validation",
                "streaming": True,
                "shuffle": False,
            },
        }
        loader = build_imagenet_dataloader(
            task_config,
            rank=0,
            world_size=1,
        )
        batch = next(iter(loader))

        self.assertEqual(set(batch), {"pixel_values", "labels"})
        self.assertEqual(tuple(batch["pixel_values"].shape), (4, 3, 224, 224))
        self.assertEqual(tuple(batch["labels"].shape), (4,))
        self.assertEqual(batch["pixel_values"].dtype, torch.float32)
        self.assertEqual(batch["labels"].dtype, torch.int64)
        self.assertTrue(torch.isfinite(batch["pixel_values"]).all().item())
        self.assertTrue(((batch["labels"] >= 0) & (batch["labels"] < 1000)).all().item())


if __name__ == "__main__":
    unittest.main()
