"""Run explicitly with torchrun --standalone --nproc-per-node=2 this_file."""
from pathlib import Path
import json
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
from salaad_moe.config import load_config
from salaad_moe.groups import StackedGroup
from salaad_moe.solver import ConsensusManager
from salaad_moe.tracking import tracking_run


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        config = load_config(Path(__file__).resolve().parents[2] / "configs/smoke_dp2.yaml")
        torch.manual_seed(3)
        # One group, two ranks: rank 1 intentionally owns no auxiliary matrices.
        parameter = torch.nn.Parameter(torch.randn(3, 4, 5))
        manager = ConsensusManager([StackedGroup("g", parameter)], config)
        manager.initialize(1)
        assert len(manager.states) == (1 if rank == 0 else 0)
        reference = manager.anchors["g"].clone()
        if rank == 0:
            with patch("salaad_moe.solver.streaming_svd_step", side_effect=RuntimeError("owner failure")):
                try:
                    manager.update(3)
                except RuntimeError as exc:
                    assert "no state committed" in str(exc)
                else:
                    raise AssertionError("Owner failure was swallowed")
        else:
            try:
                manager.update(3)
            except RuntimeError as exc:
                assert "owner failure" in str(exc)
            else:
                raise AssertionError("Failure did not reach nonowner")
        torch.testing.assert_close(manager.anchors["g"], reference, rtol=0, atol=0)
        manager.update(3)
        shards = [None] * dist.get_world_size()
        dist.all_gather_object(shards, manager.local_state_dict())
        restored = ConsensusManager([StackedGroup("g", parameter)], config)
        restored.load_shards(shards)
        torch.testing.assert_close(restored.anchors["g"], manager.anchors["g"], rtol=0, atol=0)
        invalid = manager.local_state_dict()
        if rank == 0:
            invalid["states"]["g"]["low_rank"][0, 0, 0] = float("nan")
        dist.all_gather_object(shards, invalid)
        try:
            restored.load_shards(shards)
        except RuntimeError as exc:
            assert "Invalid auxiliary checkpoint" in str(exc)
        else:
            raise AssertionError("Invalid owner state did not fail collectively")
        with patch(
            "salaad_moe.tracking.WandbTracker", side_effect=RuntimeError("logger init failed")
        ):
            try:
                with tracking_run(config, ".", "cpu"):
                    raise AssertionError("W&B failure was swallowed")
            except RuntimeError as exc:
                assert "W&B initialization" in str(exc) and "logger init failed" in str(exc)
        with patch("salaad_moe.tracking.WandbTracker") as tracker:
            with tracking_run(config, ".", "cpu"):
                pass
            assert tracker.call_count == (1 if rank == 0 else 0)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "empty_owner_rank": "passed",
                        "collective_svd_failure": "passed",
                        "owner_restore": "passed",
                        "collective_corrupt_checkpoint_failure": "passed",
                        "rank_zero_wandb_only": "passed",
                        "collective_wandb_initialization_failure": "passed",
                    }
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
