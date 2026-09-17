"""Real text packing, complete resume, serialization and likelihood contracts."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from salaad_moe.checkpoint import load_checkpoint, save_checkpoint
from salaad_moe.config import load_config
from salaad_moe.data import (
    GlobalBatchReader,
    TokenCorpus,
    build_corpus,
    documents,
    make_synthetic_corpus,
    text_split,
)
from salaad_moe.export import (
    decode_csr,
    encode_csr,
    export_checkpoint,
    export_group,
    load_evaluation_model,
    materialize_group,
    tensor_bytes,
)
from salaad_moe.model import MoELanguageModel
from salaad_moe.solver import initial_state
from salaad_moe.trainer import Trainer, evaluate_model

ROOT = Path(__file__).resolve().parents[2]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.c = load_config(ROOT / "configs/smoke.yaml")
        make_synthetic_corpus(self.c, self.path / "data", 64)
        self.corpus = TokenCorpus(self.path / "data/manifest.json", self.c)

    def tearDown(self):
        self.tmp.cleanup()

    def test_packing_shift_and_epoch_permutation(self):
        inputs, labels = self.corpus.batch("train", [0, 1], "cpu")
        torch.testing.assert_close(inputs[:, 1:], labels[:, :-1])
        self.assertEqual(inputs[1, 0].item(), labels[0, -1].item())
        reader = GlobalBatchReader(self.corpus, self.c)
        n = self.corpus.lengths["train"]
        self.assertEqual(sorted(reader.sample_index(i) for i in range(n)), list(range(n)))
        self.assertEqual(sorted(reader.sample_index(i) for i in range(n, 2 * n)), list(range(n)))
        rank0 = reader.batch(0, 0, 2, "cpu")
        rank1 = reader.batch(0, 1, 2, "cpu")
        single1 = reader.batch(1, 0, 1, "cpu")
        torch.testing.assert_close(rank1[0], single1[0])
        self.assertFalse(torch.equal(rank0[0], rank1[0]))

    def test_changed_token_file_is_rejected(self):
        path = self.path / "data/train.tokens.bin"
        with path.open("r+b") as handle:
            handle.write(b"\x00\x00\x00\x00")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            TokenCorpus(self.path / "data/manifest.json", self.c)

    def test_text_dedup_hash_split_and_tokenizer_without_model_weights(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.save(str(self.path / "tokenizer.json"))
        self.c["data"]["tokenizer_length"] = 2
        examples = {}
        for i in range(100000):
            text = f"document {i} " + "word " * 100
            split = text_split(text)[1]
            examples.setdefault(split, text)
            if len(examples) == 3:
                break
        raw = self.path / "raw.jsonl"
        texts = list(examples.values()) + [examples["train"]]
        raw.write_text("\n".join(json.dumps({"text": t}) for t in texts) + "\n")
        manifest = build_corpus(
            self.c, [raw], self.path / "text_data", self.path / "tokenizer.json"
        )
        self.assertEqual(manifest["provenance"]["duplicate_documents"], 1)
        for split in examples:
            self.assertEqual(manifest["splits"][split]["documents"], 1)
        corpus = TokenCorpus(self.path / "text_data/manifest.json", self.c)
        for split in examples:
            self.assertEqual(corpus.arrays[split][-1], 0)
        with self.assertRaises(FileExistsError):
            build_corpus(self.c, [raw], self.path / "text_data", self.path / "tokenizer.json")

    def test_zstd_jsonl_reads_concatenated_frames(self):
        import zstandard

        texts = ["first document", "A caf\u00e9 with a newline\ninside the text."]
        compressor = zstandard.ZstdCompressor()
        payload = b"".join(
            compressor.compress((json.dumps({"text": text}) + "\n\n").encode("utf-8"))
            for text in texts
        )
        for suffix in (".zst", ".zstd"):
            path = self.path / ("raw.jsonl" + suffix)
            path.write_bytes(payload)
            self.assertEqual(list(documents(path)), texts)

    def test_training_token_cap_still_fills_held_out_splits(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer_path = self.path / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        self.c["data"]["tokenizer_length"] = 2
        examples = {}
        for i in range(100000):
            text = f"document {i} " + "word " * 1200
            examples.setdefault(text_split(text)[1], text)
            if len(examples) == 3:
                break
        raw = self.path / "bounded.jsonl"
        # Place held-out documents after the training limit has already been reached.
        raw.write_text("\n".join(
            json.dumps({"text": examples[split]})
            for split in ("train", "train", "validation", "test")
        ) + "\n")
        manifest = build_corpus(
            self.c, [raw], self.path / "bounded", tokenizer_path, max_train_tokens=64
        )
        self.assertEqual(manifest["splits"]["train"]["tokens"], 65)
        self.assertEqual(manifest["splits"]["train"]["sequences"], 2)
        for split in ("validation", "test"):
            self.assertEqual(manifest["splits"][split]["sequences"], 16)
        corpus = TokenCorpus(self.path / "bounded/manifest.json", self.c)
        inputs, labels = corpus.batch("train", [1], "cpu")
        self.assertEqual(inputs.shape, (1, 32))
        torch.testing.assert_close(inputs[:, 1:], labels[:, :-1])
        with self.assertRaisesRegex(ValueError, "multiple of seq_length"):
            build_corpus(
                self.c, [raw], self.path / "invalid_cap", tokenizer_path, max_train_tokens=33
            )
        self.assertFalse((self.path / "invalid_cap").exists())

    def test_complete_checkpoint_resume_is_bitwise_identical(self):
        reference = Trainer(self.c, self.corpus, "cpu")
        self.assertTrue(reference.manager.initialized)
        self.assertEqual(reference.manager.last_structure_step, 0)
        for _ in range(4):
            reference.train_step()
        # Signed thresholds and the persistent streaming basis must survive resume.
        for state in reference.manager.states.values():
            state.tau_l.fill_(-0.002)
            state.tau_s.fill_(-0.0001)
        checkpoint = save_checkpoint(reference, self.path / "checkpoints")
        for _ in range(4):
            reference.train_step()
        resumed = Trainer(self.c, self.corpus, "cpu")
        load_checkpoint(resumed, checkpoint)
        for _ in range(4):
            resumed.train_step()
        for key, value in reference.model.state_dict().items():
            torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
        for name, state in reference.manager.states.items():
            for key, tensor in state.state_dict().items():
                torch.testing.assert_close(
                    resumed.manager.states[name].state_dict()[key], tensor, rtol=0, atol=0
                )
        for p1, p2 in zip(reference.model.parameters(), resumed.model.parameters()):
            for key, value in reference.optimizer.state[p1].items():
                torch.testing.assert_close(resumed.optimizer.state[p2][key], value, rtol=0, atol=0)
        self.assertEqual(reference.reader.state_dict(), resumed.reader.state_dict())

    def test_checkpoint_requires_complete_marker_and_same_schedule(self):
        trainer = Trainer(self.c, self.corpus, "cpu")
        trainer.train_step()
        checkpoint = save_checkpoint(trainer, self.path / "checkpoints")
        changed = copy.deepcopy(self.c)
        changed["training"]["total_optimizer_steps"] += 1
        other = Trainer(changed, self.corpus, "cpu")
        with self.assertRaisesRegex(RuntimeError, "configuration differs"):
            load_checkpoint(other, checkpoint)
        (checkpoint / "complete.json").unlink()
        with self.assertRaises(RuntimeError):
            load_checkpoint(trainer, checkpoint)

    def test_vanilla_prefix_branch_matches_salaad_from_the_start(self):
        # This optional branch scenario explicitly requests delayed initialization.
        self.c["salaad"]["state_initialization_step"] = 1
        vanilla_config = copy.deepcopy(self.c)
        vanilla_config["salaad"]["enabled"] = False
        vanilla = Trainer(vanilla_config, self.corpus, "cpu")
        vanilla.train_step()
        prefix = save_checkpoint(vanilla, self.path / "vanilla_prefix")
        branched = Trainer(self.c, self.corpus, "cpu")
        load_checkpoint(branched, prefix, branch_from_vanilla=True)
        reference = Trainer(self.c, self.corpus, "cpu")
        reference.train_step()
        for _ in range(3):
            reference.train_step()
            branched.train_step()
        for a, b in zip(reference.model.parameters(), branched.model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        changed = copy.deepcopy(self.c)
        changed["training"]["learning_rate"] *= 2
        invalid = Trainer(changed, self.corpus, "cpu")
        with self.assertRaisesRegex(RuntimeError, "preserve training"):
            load_checkpoint(invalid, prefix, branch_from_vanilla=True)

    def test_training_reads_only_training_batches(self):
        self.c["is_wandb"] = False
        trainer = Trainer(self.c, self.corpus, "cpu")
        with patch.object(self.corpus, "batch", wraps=self.corpus.batch) as batch:
            trainer.run(self.path / "run")
        self.assertTrue(batch.call_args_list)
        self.assertEqual({call.args[0] for call in batch.call_args_list}, {"train"})

    def test_bfloat16_compute_retains_float32_master_gradients(self):
        self.c["training"]["task_precision"] = "bfloat16"
        trainer = Trainer(self.c, self.corpus, "cpu")
        record = trainer.train_step()
        self.assertGreater(record["lm_nll"], 0)
        for p in trainer.model.parameters():
            self.assertEqual(p.dtype, torch.float32)
            self.assertEqual(p.grad.dtype, torch.float32)
        for state in trainer.optimizer.state.values():
            self.assertEqual(state["exp_avg"].dtype, torch.float32)

    def test_attention_recompute_matches_gradients(self):
        self.c["training"]["activation_recompute"] = "selective_attention"
        recompute = Trainer(self.c, self.corpus, "cpu")
        recompute.train_step()
        self.c["training"]["activation_recompute"] = "none"
        plain = Trainer(self.c, self.corpus, "cpu")
        plain.train_step()
        for a, b in zip(recompute.model.parameters(), plain.model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_csr_preserves_every_nonzero_and_original_precision(self):
        x = torch.tensor([[0.0, -5.0001, 1e-40], [2.0001, 0.0, 3.0001]])
        encoded = encode_csr(x)
        self.assertEqual(encoded["crow_indices"].dtype, torch.int32)
        self.assertEqual(encoded["col_indices"].dtype, torch.int32)
        self.assertEqual(encoded["values"].dtype, x.dtype)
        self.assertEqual(len(encoded["values"]), int(x.count_nonzero()))
        torch.testing.assert_close(decode_csr(encoded), x, rtol=0, atol=0)
        torch.testing.assert_close(decode_csr(encode_csr(x.double())), x.double(), rtol=0, atol=0)
        self.assertEqual(len(encode_csr(torch.zeros(3, 4))["values"]), 0)

    def test_export_preserves_full_rank_small_components_and_dense_sparse_state(self):
        state = initial_state(torch.zeros(2, 40, 40), self.c)
        state.shared.fill_(0.1234567)
        spectrum = torch.ones(40)
        spectrum[-1] = 1e-8  # Keep even components below the former numerical-rank cutoff.
        state.low_rank = torch.diag(spectrum).repeat(2, 1, 1)
        state.sparse.fill_(0.2345678)  # Density is 100%, above the former export cap.
        state.sparse[0, 0, 0] = 1e-40
        group = export_group(state)
        torch.testing.assert_close(group["shared"], state.shared, rtol=0, atol=0)
        for i, expert in enumerate(group["experts"]):
            torch.testing.assert_close(expert["low_rank"], state.low_rank[i], rtol=0, atol=0)
            torch.testing.assert_close(decode_csr(expert["sparse"]), state.sparse[i], rtol=0, atol=0)
            self.assertEqual(len(expert["sparse"]["values"]), 40 * 40)
        torch.testing.assert_close(materialize_group(group), state.reconstruction(), rtol=0, atol=0)

    def test_zero_rank_export_and_actual_artifact_materialization(self):
        x = torch.randn(3, 5, 4)
        state = initial_state(x, self.c)
        group = export_group(state)
        self.assertEqual(group["experts"][0]["low_rank"].count_nonzero(), 0)
        torch.testing.assert_close(materialize_group(group), state.reconstruction(), rtol=0, atol=0)
        self.assertGreater(tensor_bytes(group), 0)

    def test_export_checkpoint_evaluates_all_three_modes(self):
        trainer = Trainer(self.c, self.corpus, "cpu")
        for _ in range(3):
            trainer.train_step()
        checkpoint = save_checkpoint(trainer, self.path / "checkpoints")
        export = self.path / "export.pt"
        report = export_checkpoint(checkpoint, export)
        self.assertEqual(
            report["dense_model_tensor_bytes"],
            sum(p.numel() * p.element_size() for p in trainer.model.parameters()),
        )
        self.assertEqual(report["serialized_file_bytes"], export.stat().st_size)
        reconstructed, _ = load_evaluation_model(checkpoint, "reconstructed")
        exported, _ = load_evaluation_model(export, "exported")
        for name, value in reconstructed.state_dict().items():
            torch.testing.assert_close(exported.state_dict()[name], value, rtol=0, atol=0)
        raw_model, c = load_evaluation_model(checkpoint, "raw")
        for mode, path in (
            ("raw", checkpoint),
            ("reconstructed", checkpoint),
            ("exported", export),
        ):
            model, config = load_evaluation_model(path, mode)
            result = evaluate_model(
                model,
                self.corpus,
                "validation",
                3,
                2,
                config,
                torch.device("cpu"),
                distributed=False,
            )
            self.assertEqual(result["prediction_tokens"], 96)
            self.assertGreater(result["nll"], 0)
        for a, b in zip(raw_model.parameters(), trainer.model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_harness_likelihood_matches_manual_token_scoring(self):
        from salaad_moe.lm_eval_adapter import MoEHarnessLM

        model = MoELanguageModel(self.c).eval()
        adapter = MoEHarnessLM(model, None, self.c, "cpu")
        context, continuation = [1, 2, 3], [4, 5]
        value, greedy = adapter.score_tokens(context, continuation)
        logits = model(torch.tensor([[1, 2, 3, 4]])).logits[0, -2:].float()
        expected = logits.log_softmax(-1)[torch.arange(2), torch.tensor([4, 5])].sum().item()
        self.assertAlmostEqual(value, expected, places=6)
        self.assertEqual(greedy, bool(torch.equal(logits.argmax(-1), torch.tensor([4, 5]))))
        self.assertEqual(adapter.score_tokens(context, []), (0.0, True))


if __name__ == "__main__":
    unittest.main()
