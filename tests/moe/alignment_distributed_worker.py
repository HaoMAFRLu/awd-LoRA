"""torchrun --standalone --nproc-per-node=2 this_file /tmp/new-output-directory"""
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist

from salaad_moe.checkpoint import load_checkpoint, rng_state, save_checkpoint
from salaad_moe.config import load_config
from salaad_moe.data import TokenCorpus, make_synthetic_corpus
from salaad_moe.distributed import agree_or_raise
from salaad_moe.groups import StackedGroup
from salaad_moe.solver import ConsensusManager
from salaad_moe.trainer import Trainer


def equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            equal(a, b)
    else:
        assert first == second


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        current_rank = dist.get_rank()
        config = load_config(Path(__file__).resolve().parents[2] / "configs/smoke_aligned_dp2.yaml")
        torch.manual_seed(12)
        # A complete layer belongs to rank 0; rank 1 has no auxiliary state.
        groups = [
            StackedGroup(f"layers.0.moe.experts.{p}", torch.nn.Parameter(torch.randn(*shape)))
            for p, shape in (("gate", (3, 4, 5)), ("up", (3, 4, 5)), ("down", (3, 5, 4)))
        ]
        manager = ConsensusManager(groups, config)
        manager.initialize(0)
        assert len(manager.states) == (3 if current_rank == 0 else 0)
        before = manager.local_state_dict()
        anchors = {name: value.clone() for name, value in manager.anchors.items()}
        def run_failed_update():
            try:
                manager.update(4)
            except RuntimeError as exc:
                assert "no state committed" in str(exc) and "injected matching failure" in str(exc)
            else:
                raise AssertionError("Matching failure did not reach all ranks")
        if current_rank == 0:
            with patch("salaad_moe.solver.match_channels", side_effect=RuntimeError("injected matching failure")):
                run_failed_update()
        else:
            run_failed_update()
        equal(before, manager.local_state_dict())
        equal(anchors, manager.anchors)
        manager.update(4)
        shards = [None] * dist.get_world_size()
        dist.all_gather_object(shards, manager.local_state_dict())
        restored = ConsensusManager(groups, config)
        restored.load_shards(shards)
        equal(restored.anchors, manager.anchors)
        invalid = copy.deepcopy(shards)
        invalid[0]["states"]["layers.0.moe.experts.up"]["permutation"][1].zero_()
        try:
            restored.load_shards(invalid)
        except RuntimeError as exc:
            assert "bijection" in str(exc)
        else:
            raise AssertionError("Corrupt P was accepted on a DP rank")

        # Two layers now exercise different owners and real task/Adam updates.
        output = Path(sys.argv[1])
        error = None
        if current_rank == 0:
            try:
                make_synthetic_corpus(config, output / "data", 64)
            except Exception as exc:
                error = exc
        agree_or_raise(error, torch.device("cpu"), "Distributed alignment test data")
        corpus = TokenCorpus(output / "data/manifest.json", config)
        reference = Trainer(config, corpus, "cpu")
        assert set(reference.manager.states) == {
            f"layers.{current_rank}.moe.experts.{p}" for p in ("gate", "up", "down")
        }
        for _ in range(4):
            reference.train_step()
        checkpoint = save_checkpoint(reference, output / "checkpoints")
        for _ in range(4):
            reference.train_step()
        expected_rng = rng_state()
        with patch("salaad_moe.solver.initialize_alignment", side_effect=AssertionError("must load P")):
            resumed = Trainer(config, corpus, "cpu", initialize_auxiliary=False)
            load_checkpoint(resumed, checkpoint)
        for _ in range(4):
            resumed.train_step()
        equal(reference.model.state_dict(), resumed.model.state_dict())
        equal(reference.optimizer.state_dict(), resumed.optimizer.state_dict())
        equal(reference.manager.local_state_dict(), resumed.manager.local_state_dict())
        equal(reference.reader.state_dict(), resumed.reader.state_dict())
        equal(expected_rng, rng_state())
        for value in resumed.manager.anchors.values():
            other = value.clone()
            dist.broadcast(other, src=0)
            equal(other, value)
        save_checkpoint(resumed, output / "checkpoints")
        if current_rank == 0:
            result = {
                "empty_owner_rank": "passed",
                "collective_matching_failure": "passed",
                "collective_invalid_permutation": "passed",
                "joint_layer_owners": "passed",
                "replicated_native_anchors": "passed",
                "dp2_eight_step_training_and_bitwise_resume": "passed",
            }
            (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
