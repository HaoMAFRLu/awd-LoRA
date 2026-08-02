"""Configuration contract for the train_salad.py ViT task."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from salad.trainer_salad import SALADTrainer
from scripts.vit_config_generator import generate_vit_config
from scripts.vit_params import projection

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "vit_b8.yaml"
VANILLA_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "vit_b8_vanilla.yaml"
THROUGHPUT_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "vit_b8_vanilla_throughput.yaml"
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
        self.assertEqual(self.train_config["batch_size"], 96)
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
                batch_size=96,
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
        self.assertEqual(vanilla_config["num_total_iters"], 20_000)
        self.assertEqual(vanilla_config["num_freq"], 20)
        self.assertEqual(vanilla_config["save_interval"], 5_000)
        self.assertEqual(vanilla_config["batch_size"], 96)
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
                num_total_iters=20_000,
                num_freq=20,
                save_interval=5_000,
                batch_size=96,
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


if __name__ == "__main__":
    unittest.main()
