"""Configuration contract for the train_salad.py ViT task."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from salad.trainer_salad import SALADTrainer
from scripts.vit_config_generator import (
    VIT_BMIXED_ALPHA_RATE_DECAY,
    VIT_BMIXED_BETA_RATE_DECAY,
    VIT_BMIXED_BLOCK_SHAPES,
    VIT_BMIXED_SMOKE_EXCLUDED_SUFFIXES,
    VIT_MIXED_RHO_BY_SUFFIX,
    generate_vit_config,
)
from scripts.vit_params import projection

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "vit_b8.yaml"
VANILLA_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "vit_b8_vanilla.yaml"
THROUGHPUT_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "vit_b8_vanilla_throughput.yaml"
)
MIXED_RHO_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "vit_b8_all_qkv_rho5e6_fc_rho5e8.yaml"
)
BMIXED_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "vit_b8_bmixed.yaml"
BMIXED_SMOKE_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "vit_b8_bmixed_smoke.yaml"
)
MODEL_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "vit_b8_model.json"
LOCAL_PARQUET_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "salaad_vision"
    / "smoke"
    / "imagenet_val64_parquet"
)
LOCAL_CACHE_DIR = (
    REPOSITORY_ROOT / "data" / "salaad_vision" / "hf_cache_smoke" / "datasets"
)
CLUSTER_PARQUET_ROOT = Path(
    "/lustre/fast/fast/hma2/data/imagenet2012/hf_cache/hub/"
    "datasets--ILSVRC--imagenet-1k/snapshots/"
    "49e2ee26f3810fb5a7536bbf732a7b07389a47b5"
)
CLUSTER_CACHE_DIR = Path("/lustre/home/hma2/hf/datasets")


class VitB8TrainingConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with TRAIN_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            cls.train_config = yaml.safe_load(config_file)
        with VANILLA_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            cls.vanilla_config = yaml.safe_load(config_file)
        with THROUGHPUT_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            cls.throughput_config = yaml.safe_load(config_file)
        with MIXED_RHO_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            cls.mixed_rho_config = yaml.safe_load(config_file)
        with BMIXED_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            cls.bmixed_config = yaml.safe_load(config_file)
        with BMIXED_SMOKE_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            cls.bmixed_smoke_config = yaml.safe_load(config_file)
        with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            cls.model_config = json.load(config_file)

    def test_task_and_model_contract(self) -> None:
        expected_key_order = [
            "seed",
            "name",
            "model_config",
            "training_mode",
            "model_type",
            "task",
            "runtime",
            "precision",
            "num_total_iters",
            "num_freq",
            "gradient",
            "is_asyn",
            "is_init",
            "is_wandb",
            "is_monitor",
            "save_interval",
            "is_clip",
            "seed_for_shuffle",
            "batch_size",
            "num_workers",
            "data",
            "distillation",
            "scheduler",
            "optimizer",
            "layers",
        ]
        self.assertEqual(list(self.train_config), expected_key_order)
        self.assertEqual(self.train_config["name"], "vit_b8")
        self.assertEqual(self.train_config["model_type"], "dino_vitb8")
        self.assertEqual(
            self.train_config["task"],
            "dino_feature_distillation",
        )
        self.assertEqual(
            self.train_config["model_config"],
            MODEL_CONFIG_PATH.name,
        )
        self.assertEqual(
            self.train_config["distillation"]["initialization"],
            "random_init",
        )
        self.assertEqual(self.train_config["distillation"]["loss"], "mse")
        self.assertEqual(self.model_config["model_type"], "dino_vitb8")
        self.assertEqual(
            self.model_config["architectures"],
            ["DinoViTBase8"],
        )
        self.assertNotIn("student_initialization", self.model_config)
        self.assertNotIn("teacher_checkpoint_env", self.model_config)
        self.assertNotIn("teacher_checkpoint_sha256", self.model_config)

    def test_salad_cluster_data_is_selected(self) -> None:
        data = self.train_config["data"]
        self.assertEqual(self.train_config["runtime"], "cluster")
        self.assertEqual(self.train_config["training_mode"], "salad")
        self.assertEqual(self.train_config["num_total_iters"], 120_000)
        self.assertEqual(self.train_config["num_freq"], 20)
        self.assertEqual(self.train_config["save_interval"], 5_000)
        self.assertEqual(self.train_config["batch_size"], 64)
        self.assertEqual(self.train_config["num_workers"], 2)
        self.assertEqual(
            self.train_config["scheduler"]["params"]["total_steps"],
            120_000,
        )
        self.assertEqual(data["type"], "vision")
        self.assertEqual(data["location"], "cluster_snapshot")
        self.assertEqual(Path(data["root"]), CLUSTER_PARQUET_ROOT)
        self.assertEqual(Path(data["cache_dir"]), CLUSTER_CACHE_DIR)
        self.assertEqual(data["split"], "train")
        self.assertTrue(data["streaming"])
        self.assertTrue(data["shuffle"])

    def test_generator_local_defaults_use_the_parquet_slice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            generated_path = Path(temporary_directory) / "vit_b8.yaml"
            generated_config = generate_vit_config(output_path=str(generated_path))

        data = generated_config["data"]
        self.assertEqual(generated_config["runtime"], "local")
        self.assertEqual(data["location"], "local_smoke")
        self.assertEqual(Path(data["root"]), LOCAL_PARQUET_ROOT)
        self.assertEqual(Path(data["cache_dir"]), LOCAL_CACHE_DIR)
        self.assertEqual(data["split"], "validation")
        self.assertTrue(data["streaming"])
        self.assertTrue(data["shuffle"])

    def test_selected_vit_target_layers_are_student_only(self) -> None:
        suffixes = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
        expected = [
            f"backbone.blocks.{block}.{suffix}"
            for block in range(12)
            for suffix in suffixes
        ]
        layers = self.train_config["layers"]
        names = [entry["name"] for entry in layers]

        self.assertEqual(names, expected)
        self.assertEqual(len(names), len(set(names)))
        for entry in layers:
            self.assertNotIn("teacher", entry["name"])
            self.assertEqual(entry["params"]["rate_sparsity"], 0.05)
            self.assertEqual(
                entry["params"]["beta_dict"]["rate_decay"],
                0.003,
            )
            self.assertEqual(entry["params"]["rho_dict"]["rho"], 5e-6)
            self.assertNotIn("block_size", entry["params"])
            self.assertNotIn("block_sparsity", entry["params"])
            self.assertNotIn("block_p", entry["params"])
            self.assertNotIn("block_q", entry["params"])

    def test_each_vit_projection_has_an_explicit_parameter_template(self) -> None:
        params = projection()
        self.assertEqual(
            list(params),
            ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        )
        for layer_params in params.values():
            self.assertEqual(layer_params["rate_sparsity"], 0.05)
            self.assertNotIn("block_size", layer_params)
            self.assertNotIn("block_sparsity", layer_params)
            self.assertNotIn("block_p", layer_params)
            self.assertNotIn("block_q", layer_params)

    def test_generator_adds_bmixed_row_and_column_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            generated_path = Path(temporary_directory) / "vit_b8_bmixed.yaml"
            generated_config = generate_vit_config(
                name="vit_b8_bmixed",
                vit_layers=1,
                block_shape_by_suffix=VIT_BMIXED_BLOCK_SHAPES,
                output_path=str(generated_path),
            )

        by_suffix = {
            entry["name"].removeprefix("backbone.blocks.0."): entry["params"]
            for entry in generated_config["layers"]
        }
        for suffix, expected_shape in VIT_BMIXED_BLOCK_SHAPES.items():
            self.assertEqual(
                {
                    "block_p": by_suffix[suffix]["block_p"],
                    "block_q": by_suffix[suffix]["block_q"],
                },
                expected_shape,
            )
        self.assertEqual(
            set(by_suffix),
            {"attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"},
        )
        for suffix in {"mlp.fc1", "mlp.fc2"}:
            self.assertNotIn("block_p", by_suffix[suffix])
            self.assertNotIn("block_q", by_suffix[suffix])

    def test_generator_rejects_invalid_block_shape_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = str(Path(temporary_directory) / "invalid.yaml")
            with self.assertRaisesRegex(ValueError, "unknown block-shape"):
                generate_vit_config(
                    block_shape_by_suffix={
                        "unknown": {"block_p": 1, "block_q": "full"}
                    },
                    output_path=output_path,
                )
            with self.assertRaisesRegex(ValueError, "is missing"):
                generate_vit_config(
                    block_shape_by_suffix={"attn.qkv": {"block_p": 1}},
                    output_path=output_path,
                )
            with self.assertRaisesRegex(ValueError, "must be positive"):
                generate_vit_config(
                    block_shape_by_suffix={
                        "mlp.fc1": {"block_p": 0, "block_q": "full"}
                    },
                    output_path=output_path,
                )
            with self.assertRaisesRegex(ValueError, "unknown excluded"):
                generate_vit_config(
                    excluded_suffixes=("unknown",),
                    output_path=output_path,
                )

    def test_committed_bmixed_config_structures_attention_only(self) -> None:
        config = self.bmixed_config
        self.assertEqual(config["name"], "vit_b8_bmixed")
        self.assertEqual(
            [entry["name"] for entry in config["layers"]],
            [entry["name"] for entry in self.train_config["layers"]],
        )

        counts = {
            (1, "full"): 0,
            ("full", 1): 0,
            (None, None): 0,
        }
        for entry in config["layers"]:
            suffix = entry["name"].split(".", 3)[-1]
            actual_shape = {
                "block_p": entry["params"].get("block_p"),
                "block_q": entry["params"].get("block_q"),
            }
            expected_shape = VIT_BMIXED_BLOCK_SHAPES.get(
                suffix,
                {"block_p": None, "block_q": None},
            )
            self.assertEqual(actual_shape, expected_shape, suffix)
            self.assertEqual(
                entry["params"]["rho_dict"]["rho"],
                VIT_MIXED_RHO_BY_SUFFIX[suffix],
                suffix,
            )
            self.assertEqual(
                entry["params"]["alpha_dict"]["rate_decay"],
                VIT_BMIXED_ALPHA_RATE_DECAY,
                suffix,
            )
            self.assertEqual(
                entry["params"]["beta_dict"]["rate_decay"],
                VIT_BMIXED_BETA_RATE_DECAY,
                suffix,
            )
            counts[(actual_shape["block_p"], actual_shape["block_q"])] += 1
        self.assertEqual(
            counts,
            {(1, "full"): 12, ("full", 1): 12, (None, None): 24},
        )

        for key, value in self.train_config.items():
            if key not in {"name", "layers"}:
                self.assertEqual(config[key], value, key)

    def test_generator_reproduces_committed_bmixed_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            generated_path = Path(temporary_directory) / BMIXED_CONFIG_PATH.name
            generate_vit_config(
                name="vit_b8_bmixed",
                lr=1e-4,
                num_total_iters=120_000,
                num_freq=20,
                save_interval=5_000,
                batch_size=64,
                warmup_steps=2_000,
                scheduler_total_steps=120_000,
                num_workers=2,
                runtime="cluster",
                data_location="cluster_snapshot",
                data_root=str(CLUSTER_PARQUET_ROOT),
                data_cache_dir=str(CLUSTER_CACHE_DIR),
                data_split="train",
                vit_layers=-1,
                rho_by_suffix=VIT_MIXED_RHO_BY_SUFFIX,
                block_shape_by_suffix=VIT_BMIXED_BLOCK_SHAPES,
                alpha_rate_decay=VIT_BMIXED_ALPHA_RATE_DECAY,
                beta_rate_decay=VIT_BMIXED_BETA_RATE_DECAY,
                output_path=str(generated_path),
            )
            with generated_path.open("r", encoding="utf-8") as config_file:
                generated_config = yaml.safe_load(config_file)

        self.assertEqual(generated_config, self.bmixed_config)

    def test_bmixed_smoke_config_is_short_local_and_attention_structured(
        self,
    ) -> None:
        config = self.bmixed_smoke_config
        self.assertEqual(config["name"], "vit_b8_bmixed_smoke")
        self.assertEqual(config["runtime"], "local")
        self.assertEqual(config["num_total_iters"], 10)
        self.assertEqual(config["num_freq"], 2)
        self.assertEqual(config["save_interval"], 2)
        self.assertEqual(config["batch_size"], 1)
        self.assertEqual(config["num_workers"], 0)
        self.assertFalse(config["is_wandb"])

        data = config["data"]
        self.assertEqual(data["location"], "local_smoke")
        self.assertEqual(data["split"], "validation")
        self.assertEqual(
            Path(data["root"]),
            Path("data/salaad_vision/smoke/imagenet_val64_parquet"),
        )
        self.assertEqual(
            Path(data["cache_dir"]),
            Path("data/salaad_vision/hf_cache_smoke/datasets"),
        )
        self.assertTrue(data["streaming"])
        self.assertFalse(data["shuffle"])

        by_suffix = {
            entry["name"].removeprefix("backbone.blocks.0."): entry["params"]
            for entry in config["layers"]
        }
        self.assertEqual(
            set(by_suffix),
            {"attn.qkv"},
        )
        self.assertEqual(
            {
                "block_p": by_suffix["attn.qkv"]["block_p"],
                "block_q": by_suffix["attn.qkv"]["block_q"],
            },
            {"block_p": 1, "block_q": "full"},
        )
        self.assertEqual(
            by_suffix["attn.qkv"]["alpha_dict"]["rate_decay"],
            VIT_BMIXED_ALPHA_RATE_DECAY,
        )
        self.assertEqual(
            by_suffix["attn.qkv"]["beta_dict"]["rate_decay"],
            VIT_BMIXED_BETA_RATE_DECAY,
        )

    def test_generator_reproduces_committed_bmixed_smoke_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            generated_path = (
                Path(temporary_directory) / BMIXED_SMOKE_CONFIG_PATH.name
            )
            generate_vit_config(
                name="vit_b8_bmixed_smoke",
                model_config="vit_b8_model.json",
                training_mode="salad",
                lr=1e-5,
                num_freq=2,
                is_wandb=False,
                is_monitor=True,
                save_interval=2,
                num_total_iters=10,
                batch_size=1,
                warmup_steps=1,
                scheduler_total_steps=10,
                num_workers=0,
                precision="bfloat16",
                runtime="local",
                data_location="local_smoke",
                data_root="data/salaad_vision/smoke/imagenet_val64_parquet",
                data_cache_dir="data/salaad_vision/hf_cache_smoke/datasets",
                data_split="validation",
                data_streaming=True,
                data_shuffle=False,
                shuffle_buffer_size=64,
                distillation_initialization="random_init",
                include_attention=True,
                include_mlp=True,
                vit_layers=1,
                block_shape_by_suffix=VIT_BMIXED_BLOCK_SHAPES,
                excluded_suffixes=VIT_BMIXED_SMOKE_EXCLUDED_SUFFIXES,
                alpha_rate_decay=VIT_BMIXED_ALPHA_RATE_DECAY,
                beta_rate_decay=VIT_BMIXED_BETA_RATE_DECAY,
                output_path=str(generated_path),
            )
            with generated_path.open("r", encoding="utf-8") as config_file:
                generated_config = yaml.safe_load(config_file)

        self.assertEqual(generated_config, self.bmixed_smoke_config)

    def test_all_target_layers_are_evenly_owned_by_four_ranks(self) -> None:
        expected_names = {entry["name"] for entry in self.train_config["layers"]}
        owned_names = set()

        for rank in range(4):
            assigned, owner_map = SALADTrainer.assign_layers(
                self.train_config["layers"],
                rank,
                4,
            )
            self.assertEqual(len(assigned), 12)
            self.assertTrue(owned_names.isdisjoint(assigned))
            self.assertTrue(all(owner_map[name] == rank for name in assigned))
            owned_names.update(assigned)

        self.assertEqual(owned_names, expected_names)

    def test_generator_reproduces_committed_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            generated_path = Path(temporary_directory) / "vit_b8.yaml"
            generate_vit_config(
                lr=1e-4,
                num_total_iters=120_000,
                num_freq=20,
                save_interval=5_000,
                batch_size=64,
                warmup_steps=2_000,
                num_workers=2,
                runtime="cluster",
                data_location="cluster_snapshot",
                data_root=str(CLUSTER_PARQUET_ROOT),
                data_cache_dir=str(CLUSTER_CACHE_DIR),
                data_split="train",
                vit_layers=-1,
                output_path=str(generated_path),
            )
            with generated_path.open("r", encoding="utf-8") as config_file:
                generated_config = yaml.safe_load(config_file)

        self.assertEqual(generated_config, self.train_config)

    def test_salad_and_vanilla_share_non_layer_hyperparameters(self) -> None:
        mode_specific_keys = {
            "name",
            "training_mode",
            "num_total_iters",
            "layers",
        }
        salad_common = {
            key: value
            for key, value in self.train_config.items()
            if key not in mode_specific_keys
        }
        vanilla_common = {
            key: value
            for key, value in self.vanilla_config.items()
            if key not in mode_specific_keys
        }

        self.assertEqual(salad_common, vanilla_common)

    def test_vanilla_cluster_training_contract(self) -> None:
        vanilla_config = self.vanilla_config
        data = vanilla_config["data"]

        self.assertEqual(vanilla_config["name"], "vit_b8_vanilla")
        self.assertEqual(vanilla_config["training_mode"], "vanilla")
        self.assertEqual(vanilla_config["layers"], [])
        self.assertEqual(vanilla_config["runtime"], "cluster")
        self.assertEqual(vanilla_config["num_total_iters"], 120_000)
        self.assertEqual(vanilla_config["num_freq"], 20)
        self.assertEqual(vanilla_config["save_interval"], 5_000)
        self.assertEqual(vanilla_config["batch_size"], 64)
        self.assertEqual(vanilla_config["num_workers"], 2)
        self.assertEqual(
            vanilla_config["scheduler"]["params"]["total_steps"],
            120_000,
        )
        self.assertEqual(data["location"], "cluster_snapshot")
        self.assertEqual(Path(data["root"]), CLUSTER_PARQUET_ROOT)
        self.assertEqual(Path(data["cache_dir"]), CLUSTER_CACHE_DIR)
        self.assertEqual(data["split"], "train")
        self.assertTrue(data["streaming"])
        self.assertTrue(data["shuffle"])
        self.assertEqual(
            vanilla_config["distillation"],
            self.train_config["distillation"],
        )

    def test_generator_reproduces_committed_vanilla_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            generated_path = Path(temporary_directory) / "vit_b8_vanilla.yaml"
            generate_vit_config(
                name="vit_b8_vanilla",
                training_mode="vanilla",
                num_total_iters=120_000,
                num_freq=20,
                save_interval=5_000,
                batch_size=64,
                lr=1e-4,
                warmup_steps=2_000,
                scheduler_total_steps=120_000,
                num_workers=2,
                runtime="cluster",
                data_location="cluster_snapshot",
                data_root=str(CLUSTER_PARQUET_ROOT),
                data_cache_dir=str(CLUSTER_CACHE_DIR),
                data_split="train",
                output_path=str(generated_path),
            )
            with generated_path.open("r", encoding="utf-8") as config_file:
                generated_config = yaml.safe_load(config_file)

        self.assertEqual(generated_config, self.vanilla_config)

    def test_throughput_config_only_shortens_the_run(self) -> None:
        expected = dict(self.vanilla_config)
        expected.update(
            name="vit_b8_vanilla_throughput",
            num_total_iters=400,
            num_freq=10,
            save_interval=400,
        )
        self.assertEqual(self.throughput_config, expected)

    def test_mixed_rho_config_only_weakens_non_qkv_layers(self) -> None:
        config = self.mixed_rho_config
        self.assertEqual(
            config["name"],
            "vit_b8_all_qkv_rho5e6_fc_rho5e8",
        )
        self.assertEqual(
            [entry["name"] for entry in config["layers"]],
            [entry["name"] for entry in self.train_config["layers"]],
        )

        counts = {5e-6: 0, 5e-8: 0}
        for entry in config["layers"]:
            name = entry["name"]
            rho = entry["params"]["rho_dict"]["rho"]
            expected = 5e-6 if name.endswith("attn.qkv") else 5e-8
            self.assertEqual(rho, expected, name)
            counts[rho] += 1
        self.assertEqual(counts, {5e-6: 12, 5e-8: 36})

        for key, value in self.train_config.items():
            if key not in {"name", "layers"}:
                self.assertEqual(config[key], value, key)

    def test_generator_reproduces_committed_mixed_rho_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            generated_path = Path(temporary_directory) / MIXED_RHO_CONFIG_PATH.name
            generate_vit_config(
                name="vit_b8_all_qkv_rho5e6_fc_rho5e8",
                lr=1e-4,
                num_total_iters=120_000,
                num_freq=20,
                save_interval=5_000,
                batch_size=64,
                warmup_steps=2_000,
                scheduler_total_steps=120_000,
                num_workers=2,
                runtime="cluster",
                data_location="cluster_snapshot",
                data_root=str(CLUSTER_PARQUET_ROOT),
                data_cache_dir=str(CLUSTER_CACHE_DIR),
                data_split="train",
                vit_layers=-1,
                rho_by_suffix={
                    "attn.qkv": 5e-6,
                    "attn.proj": 5e-8,
                    "mlp.fc1": 5e-8,
                    "mlp.fc2": 5e-8,
                },
                output_path=str(generated_path),
            )
            with generated_path.open("r", encoding="utf-8") as config_file:
                generated_config = yaml.safe_load(config_file)

        self.assertEqual(generated_config, self.mixed_rho_config)

    def test_generator_rejects_invalid_rho_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = str(Path(temporary_directory) / "invalid.yaml")
            with self.assertRaisesRegex(ValueError, "unknown rho override"):
                generate_vit_config(
                    rho_by_suffix={"unknown": 5e-8},
                    output_path=output_path,
                )
            with self.assertRaisesRegex(ValueError, "must be positive"):
                generate_vit_config(
                    rho_by_suffix={"mlp.fc1": 0.0},
                    output_path=output_path,
                )


if __name__ == "__main__":
    unittest.main()
