#!/usr/bin/env bash
# Save the first soft P_e update, then inspect its direct Hungarian projection.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 OUTPUT_DIRECTORY" >&2
    exit 2
fi

run_dir="$1"
python_bin="/lustre/home/hma2/projects/awd-LoRA/myenv/bin/python"
export WANDB_MODE=disabled

"$python_bin" -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_salad.py \
    --cfg_version ns97m_sinkhorn_10k_first_update \
    --device cuda --cpu-threads 4 --stop-after 10 \
    --data-manifest /lustre/home/hma2/projects/awd-LoRA/data/moe_corpora/dclm_20260916/tokens_ns97m/manifest.json \
    --output "$run_dir"

"$python_bin" - "$run_dir" <<'PY'
import json
from pathlib import Path
import sys
from salaad_moe.checkpoint import checkpoint_metadata, load_torch

run = Path(sys.argv[1])
checkpoint = run / "checkpoints/step_00000010"
meta = checkpoint_metadata(checkpoint)
assert meta["step"] == 10 and meta["world_size"] == 4
config = json.loads((run / "config.resolved.json").read_text())
assert config["is_wandb"] is False
assert not (run / "wandb_run.json").exists() and not (run / "wandb").exists()
records = []
for rank in range(meta["world_size"]):
    state = load_torch(checkpoint / f"rank_{rank:05d}.pt")["salaad"]
    record = {"rank": rank, **{key: state[key] for key in
              ("sweeps", "last_structure_step", "last_matching_step")}}
    assert record["sweeps"] == 1, record
    assert record["last_structure_step"] == record["last_matching_step"] == 10, record
    records.append(record)
report = {"checkpoint": str(checkpoint), **meta, "wandb_enabled": False, "ranks": records}
(run / "first_pe_verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
PY

"$python_bin" -u scripts/analyze_moe_pe_row_maxima.py \
    --checkpoint "$run_dir/checkpoints/step_00000010" \
    --output "$run_dir/first_pe_hungarian" \
    --expected-step 10 --project-hungarian --hungarian-tie-rule solver --cpu-threads 4
