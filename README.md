# awd-LoRA

Research code for training and evaluating low-rank/sparse decompositions of
language-model weights. The current codebase centers on **SALAD**, an
ADMM-style method that trains LLaMA-style causal language models while
encouraging selected weight matrices to decompose as:

```text
X ~= L + S
```

where `X` is the trainable model weight, `L` is a low-rank component, and `S`
is a sparse component.

The repository contains distributed training code, model definitions,
configuration generation, evaluation utilities, plotting/analysis scripts, and
cluster submit examples used for experiments.

## Repository Layout

```text
salad/              Core SALAD trainer, solver, operators, and utilities
models/             Local model definitions, including a LLaMA implementation
dataloaders/        Iterable dataset wrappers and tokenization utilities
scripts/            Training, evaluation, config generation, plotting, analysis
configs/            YAML training configs and model JSON configs
analysis/           One-off analysis scripts used during experiments
sub/                HTCondor submit-file examples for cluster runs
utils/              Additional utility code
```

The main training path is:

```text
scripts/train_salad.py
  -> salad/register.py
  -> salad/trainer_salad.py
  -> salad/salad_solver.py
```

## Installation

This project expects Python 3.9+ and a CUDA-enabled PyTorch installation for
real training runs. The pinned dependencies in `requirements.txt` reflect the
development environment used for experiments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Training and evaluation use Hugging Face datasets/models. If you hit rate limits
or need access to gated resources, log in before running experiments:

```bash
huggingface-cli login
```

Weights & Biases logging is controlled by config values such as `is_wandb`.
If enabled, set:

```bash
export WANDB_API_KEY=...
```

## Configuration

Training configs live in `configs/*.yaml`; matching model architecture
configs live in `configs/*_model.json`.

Important fields:

- `training_mode`: `salad` for the original decomposition, `consensus` for the
  loop-specific consensus formulation, or `vanilla` for normal training.
- `num_total_iters`: total optimizer updates.
- `num_freq`: how often to run SALAD/ADMM updates.
- `batch_size`, `max_length`, `num_workers`: data-loading and tokenization
  settings.
- `optimizer`, `scheduler`: optimizer and learning-rate schedule.
- `layers`: model layers to decompose.
- `rate_rank`: target low-rank ratio for a layer.
- `rate_sparsity`: target sparse density for a layer.
- `rho_dict`, `alpha_dict`, `beta_dict`: ADMM penalty and adaptive threshold
  settings.
- `loop.sampling`: distributed sampling policy for the logical loop count.
- `consensus_salaad`: consensus `rho`, nuclear-norm coefficient
  `lambda_low_rank`, and l1 coefficient `lambda_sparse`.

To regenerate configs, edit `scripts/config_generator.py` and run:

```bash
python scripts/config_generator.py
```

## Training

The default entry point is distributed and expects GPUs:

```bash
torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29500 \
  scripts/train_salad.py \
  --cfg_version llama_consensus_60m
```

For multi-GPU training, increase `--nproc_per_node`. The trainer wraps the model
with PyTorch DistributedDataParallel and assigns configured layers across ranks.

Select a config with `--cfg_version`; `--folder` optionally overrides its
`output_folder` value.

Example HTCondor submit files are provided in `sub/`. They contain local cluster
paths and hardware constraints from the original experiment environment, so they
should be treated as templates rather than portable launch scripts.

## Outputs

Training writes experiment artifacts under:

```text
data/<folder>/<cfg_version>/<timestamp>/
```

Typical outputs:

- `model.pth`: trained model checkpoint.
- `<cfg_version>.yaml`: copied training config.
- `<cfg_version>_model.json`: copied model config.
- `layer_info.pkl`: loss, rank, sparsity, rho, alpha, and beta traces.
- `matrix_rank<N>.pkl`: rank-local low-rank/sparse/dual variables.
- `consensus_rank<N>.pth`: rank-local shared, low-rank, sparse, and scaled-dual
  variables for Consensus SALAAD.

Generated data, caches, W&B runs, and checkpoints can be large and are not
required for understanding the source code.

## Evaluation

The repository includes several evaluation scripts:

- `scripts/evaluation.py`: evaluates trained SALAD models by reconstructing
  selected layers from saved low-rank and sparse components.
- `scripts/multi_evaluation.py` and `scripts/multi_ddp_evaluation.py`: batch and
  distributed variants used in experiments.
- `scripts/resave_model.py`: converts saved checkpoints into Hugging Face-style
  folders for downstream evaluation.
- `scripts/run_lm_eval.py`: runs LM Evaluation Harness tasks such as PIQA,
  Winogrande, ARC, BoolQ, MMLU, HellaSwag, GSM8K, and TruthfulQA.

Some evaluation and analysis scripts are experiment-specific and may require
path/config edits before running outside the original environment.

## Method Overview

During training, the model optimizes the causal-language-model loss. In SALAD
mode, configured layers also receive an ADMM penalty:

```text
rho / 2 * ||X - L - S + Y / rho||_F^2
```

Every `num_freq` iterations, each assigned layer updates:

- `L` with SVD and singular-value thresholding.
- `S` with elementwise soft thresholding.
- `Y` with the ADMM dual update.
- `alpha`, `beta`, and optionally `rho` according to their configured adaptive
  rules.

The saved `L` and `S` matrices can later be recombined or further truncated to
study parameter-count/perplexity tradeoffs.

In Consensus SALAAD, the recurrent body uses a dense matrix `X_i` for logical
loop `i`, while ADMM enforces:

```text
X_i = X + L_i + S_i
```

Here `X` is shared by every loop, and `L_i` and `S_i` are loop-specific
low-rank and sparse residuals. The task optimizer updates the dense `X_i`
variables through both the language-model loss and the augmented penalty.
Every `num_freq` iterations, the solver updates the shared mean `X`, applies
singular-value and elementwise thresholding to `L_i` and `S_i`, and updates the
scaled dual. `salad.consensus.apply_decomposition` materializes `X + L_i + S_i`
into the model for evaluation.

## Notes for Contributors

This is research code. Some scripts are intentionally experimental, and not all
entry points are guaranteed to be portable without editing paths and configs.
Before contributing or running large experiments:

- Start with a small config such as `llama_debug`.
- Verify that Hugging Face dataset streaming works in your environment.
- Disable W&B logging unless you have configured credentials.
- Treat files in `sub/` as examples for a specific cluster setup.

## License

This repository is released under the license in `LICENSE`.
