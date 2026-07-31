"""Configuration contract for the train_salad.py ViT task."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from scripts.vit_config_generator import generate_vit_config
from scripts.vit_params import projection

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "vit_b8.yaml"
VANILLA_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "vit_b8_vanilla.yaml"
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


class VitB8TrainingConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with TRAIN_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            cls.train_config = yaml.safe_load(config_file)
        with VANILLA_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            cls.vanilla_config = yaml.safe_load(config_file)
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

    def test_local_parquet_data_is_selected(self) -> None:
        data = self.train_config["data"]
        self.assertEqual(self.train_config["runtime"], "local")
        self.assertEqual(data["type"], "vision")
        self.assertEqual(data["location"], "local_smoke")
        self.assertEqual(Path(data["root"]), LOCAL_PARQUET_ROOT)
        self.assertEqual(Path(data["cache_dir"]), LOCAL_CACHE_DIR)
        self.assertEqual(data["split"], "validation")
        self.assertTrue(data["streaming"])
        self.assertFalse(data["shuffle"])

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
        self.assertFalse(data["shuffle"])

    def test_selected_vit_target_layers_are_student_only(self) -> None:
        suffixes = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
        expected = [
            f"backbone.blocks.{block}.{suffix}"
            for block in range(1)
            for suffix in suffixes
        ]
        layers = self.train_config["layers"]
        names = [entry["name"] for entry in layers]

        self.assertEqual(names, expected)
        self.assertEqual(len(names), len(set(names)))
        for entry in layers:
            self.assertNotIn("teacher", entry["name"])
            self.assertEqual(entry["params"]["rate_sparsity"], 0.05)
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

    def test_generator_reproduces_committed_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            generated_path = Path(temporary_directory) / "vit_b8.yaml"
            generate_vit_config(output_path=str(generated_path))
            with generated_path.open("r", encoding="utf-8") as config_file:
                generated_config = yaml.safe_load(config_file)

        self.assertEqual(generated_config, self.train_config)

    def test_vanilla_config_only_disables_salaad(self) -> None:
        salad_config = dict(self.train_config)
        vanilla_config = dict(self.vanilla_config)

        self.assertEqual(vanilla_config["name"], "vit_b8_vanilla")
        self.assertEqual(vanilla_config["training_mode"], "vanilla")
        self.assertEqual(vanilla_config["layers"], [])

        salad_config.pop("name")
        salad_config.pop("training_mode")
        salad_config.pop("layers")
        vanilla_config.pop("name")
        vanilla_config.pop("training_mode")
        vanilla_config.pop("layers")
        self.assertEqual(vanilla_config, salad_config)

    def test_generator_reproduces_committed_vanilla_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            generated_path = Path(temporary_directory) / "vit_b8_vanilla.yaml"
            generate_vit_config(
                name="vit_b8_vanilla",
                training_mode="vanilla",
                output_path=str(generated_path),
            )
            with generated_path.open("r", encoding="utf-8") as config_file:
                generated_config = yaml.safe_load(config_file)

        self.assertEqual(generated_config, self.vanilla_config)


if __name__ == "__main__":
    unittest.main()
