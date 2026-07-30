"""Tests for explicit local and cluster ImageNet loader configuration."""

from __future__ import annotations

import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from salaad_vision.data.imagenet import (
    CLUSTER_DATASETS_CACHE_ENV,
    CLUSTER_IMAGENET_ROOT_ENV,
    DEFAULT_CLUSTER_DATASETS_CACHE,
    DEFAULT_CLUSTER_IMAGENET_ROOT,
    DEFAULT_LOCAL_IMAGENET_ROOT,
    LOCAL_IMAGENET_ROOT_ENV,
    ImageNetDataLocation,
    ImageNetLoaderConfig,
    build_imagenet_dataloader,
)


class ImageNetDataLoaderTest(unittest.TestCase):
    def test_local_and_cluster_paths_are_explicitly_distinct(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(LOCAL_IMAGENET_ROOT_ENV, None)
            os.environ.pop(CLUSTER_IMAGENET_ROOT_ENV, None)
            os.environ.pop(CLUSTER_DATASETS_CACHE_ENV, None)
            os.environ.pop("IMAGENET_SMOKE_ROOT", None)
            local = ImageNetLoaderConfig.local_smoke()
            cluster = ImageNetLoaderConfig.cluster_snapshot()

        self.assertEqual(local.location, ImageNetDataLocation.LOCAL_SMOKE)
        self.assertEqual(local.root, DEFAULT_LOCAL_IMAGENET_ROOT)
        self.assertEqual(local.split, "validation")
        self.assertFalse(local.shuffle)

        self.assertEqual(cluster.location, ImageNetDataLocation.CLUSTER_SNAPSHOT)
        self.assertEqual(cluster.root, DEFAULT_CLUSTER_IMAGENET_ROOT)
        self.assertEqual(cluster.cache_dir, DEFAULT_CLUSTER_DATASETS_CACHE)
        self.assertEqual(cluster.split, "train")
        self.assertTrue(cluster.shuffle)
        self.assertTrue(cluster.drop_last)
        self.assertNotEqual(local.root, cluster.root)

        cluster_validation = ImageNetLoaderConfig.cluster_snapshot(
            split="validation"
        )
        self.assertFalse(cluster_validation.shuffle)
        self.assertFalse(cluster_validation.drop_last)

    def test_environment_overrides_do_not_cross_locations(self) -> None:
        environment = {
            LOCAL_IMAGENET_ROOT_ENV: "/tmp/local-imagenet-smoke",
            CLUSTER_IMAGENET_ROOT_ENV: "/tmp/cluster-imagenet-snapshot",
            CLUSTER_DATASETS_CACHE_ENV: "/tmp/cluster-imagenet-cache",
        }
        with patch.dict(os.environ, environment, clear=False):
            local = ImageNetLoaderConfig.local_smoke()
            cluster = ImageNetLoaderConfig.cluster_snapshot()

        self.assertEqual(str(local.root), environment[LOCAL_IMAGENET_ROOT_ENV])
        self.assertEqual(str(cluster.root), environment[CLUSTER_IMAGENET_ROOT_ENV])
        self.assertEqual(
            str(cluster.cache_dir),
            environment[CLUSTER_DATASETS_CACHE_ENV],
        )

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
            config = ImageNetLoaderConfig.cluster_snapshot(
                root=temporary_root,
                split="validation",
                batch_size=1,
                num_workers=0,
                cache_dir=Path(temporary_root) / "cache",
                pin_memory=False,
            )
            batch = next(iter(build_imagenet_dataloader(config)))

        self.assertEqual(tuple(batch["pixel_values"].shape), (1, 3, 224, 224))
        self.assertEqual(batch["labels"].tolist(), [17])

    def test_local_smoke_batch_contract(self) -> None:
        config = ImageNetLoaderConfig.local_smoke(batch_size=4)
        if not config.root.is_dir():
            self.skipTest(f"local ImageNet smoke set is absent: {config.root}")

        loader = build_imagenet_dataloader(config)
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
