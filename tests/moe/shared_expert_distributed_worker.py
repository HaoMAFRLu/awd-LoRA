"""torchrun --standalone --nproc-per-node=2 this_file /tmp/new-output-directory"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist

from salaad_moe.checkpoint import load_checkpoint, save_checkpoint
from salaad_moe.config import load_config
from salaad_moe.data import TokenCorpus, make_synthetic_corpus
from salaad_moe.distributed import agree_or_raise
from salaad_moe.trainer import Trainer


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        root = Path(sys.argv[1])
        for topk in (2, 3):
            config = load_config(Path(__file__).resolve().parents[2] / "configs/smoke_dp2.yaml")
            config["model"].update(
                family="llama_style_shared_expert", num_experts=7,
                num_shared_experts=1, router_topk=topk,
            )
            config["salaad"]["enabled"] = False
            config["is_wandb"] = False
            config["training"]["task_precision"] = "bfloat16"
            output = root / f"topk_{topk}"
            error = None
            if dist.get_rank() == 0:
                try:
                    make_synthetic_corpus(config, output / "data", 64)
                except Exception as exc:
                    error = exc
            agree_or_raise(error, torch.device("cpu"), "Prepare shared-expert test corpus")
            corpus = TokenCorpus(output / "data/manifest.json", config)
            reference = Trainer(config, corpus, "cpu")
            assert reference.manager is None
            for _ in range(2):
                reference.train_step()
            checkpoint = save_checkpoint(reference, output / "checkpoints")
            for _ in range(2):
                reference.train_step()
            resumed = Trainer(config, corpus, "cpu")
            load_checkpoint(resumed, checkpoint)
            for _ in range(2):
                resumed.train_step()
            for name, value in reference.model.state_dict().items():
                torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
            for p, q in zip(reference.model.parameters(), resumed.model.parameters()):
                for key, value in reference.optimizer.state[p].items():
                    torch.testing.assert_close(value, resumed.optimizer.state[q][key], rtol=0, atol=0)
            assert reference.reader.state_dict() == resumed.reader.state_dict()
            flattened = torch.cat([p.detach().reshape(-1) for p in resumed.model.parameters()])
            replicas = [torch.empty_like(flattened) for _ in range(dist.get_world_size())]
            dist.all_gather(replicas, flattened)
            for replica in replicas:
                torch.testing.assert_close(replica, flattened, rtol=0, atol=0)
            if dist.get_rank() == 0:
                print(json.dumps({"shared_experts": 1, "router_topk": topk,
                                  "dp2_bfloat16_training_bitwise_resume_and_replicas": "passed"}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
