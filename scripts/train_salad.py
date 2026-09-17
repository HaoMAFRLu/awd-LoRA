"""MoE training entry point with SALAAD consensus, low-rank, and sparse constraints.

Reading order: main() in this file -> Trainer.train_step() -> ConsensusManager.
Single-GPU debugging with the local DCLM slice: myenv/bin/python scripts/train_salad.py
For multiprocess data parallelism (DP), launch this same script with torchrun.
"""
import argparse
import json
import os
from pathlib import Path
import sys

root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)

import torch.distributed as dist

from salaad_moe.config import config_for_version, load_config, parameter_counts, validate_config
from salaad_moe.distributed import world_size
from salaad_moe.runtime import initialize_device, prepare_corpus, resolve_output_directory
from salaad_moe.trainer import Trainer


LOCAL_SMOKE_MANIFEST = str(
    Path(root) / 'data/moe_corpora/dclm_20260916/smoke_tokens/manifest.json'
)


def parse_args(argv=None):
    """Use the local real-text smoke test when the IDE supplies no arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--cfg_version', default='smoke_dclm_bf16', help='Config name in configs/')
    source.add_argument('--config', default=None, help='Use configs/<cfg_version>.yaml unless overridden')
    parser.add_argument('--folder', default='salaad_moe', help='Output group under data/')
    # A new timestamped directory keeps repeated debug runs separate.
    parser.add_argument(
        '--output', default=None,
        help='Use data/<folder>/<config>/<timestamp>/ unless overridden',
    )
    parser.add_argument(
        '--data-manifest', default=None,
        help='Prepared token corpus manifest.json; defaults to the local slice for smoke_dclm_bf16',
    )
    # None means use the selected YAML: rho=0.05 and eight steps for local smoke.
    parser.add_argument(
        '--rho', type=float, default=None,
        help='Use the YAML penalty coefficient (local smoke: 0.05) unless overridden',
    )
    parser.add_argument(
        '--num_total_iters', '--stop-after', dest='num_total_iters', type=int, default=None,
        help='Run the full YAML schedule (local smoke: 8 steps), or pause at the specified step',
    )
    # Local debugging starts a fresh run; checkpoint paths are optional overrides.
    restore = parser.add_mutually_exclusive_group()
    restore.add_argument('--resume', default=None, help='Complete checkpoint directory to resume')
    restore.add_argument('--branch-from', default=None, help='Vanilla prefix checkpoint for a SALAAD branch')
    parser.add_argument(
        '--device', choices=('cpu', 'cuda'), default=None,
        help='Automatically use CUDA when available, otherwise CPU',
    )
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--allow-synthetic', action='store_true', default=False, help='Allow explicit smoke data')
    parser.add_argument('--dry-run', action='store_true', default=False, help='Print config/counts without training')
    args = parser.parse_args(argv)
    # Default to the local DCLM slice without overriding other configs or resumed runs.
    if (
        args.data_manifest is None
        and args.config is None
        and args.cfg_version.removesuffix('.yaml') == 'smoke_dclm_bf16'
        and not args.resume
        and not args.branch_from
    ):
        args.data_manifest = LOCAL_SMOKE_MANIFEST
    return args


def main(cfg_version='smoke_dclm_bf16', path_cfg=None, folder='salaad_moe',
         num_total_iters=None, rho=None, *, data_manifest=None, output=None,
         resume=None, branch_from=None, device=None, cpu_threads=4,
         allow_synthetic=False, dry_run=False):
    """Start all MoE training here; set a common breakpoint on the first statement."""
    # 1. The YAML defines the MoE model, training budget, and SALAAD settings.
    # No separate *_model.json is needed.
    path_cfg = Path(path_cfg) if path_cfg is not None else config_for_version(cfg_version)
    config = load_config(path_cfg)
    if rho is not None:
        config['salaad']['rho'] = rho
    validate_config(config)
    if num_total_iters is not None and num_total_iters < 1:
        raise ValueError('--num_total_iters/--stop-after must be positive')
    if resume and branch_from:
        raise ValueError('Choose either resume or branch_from')
    if dry_run:
        print(json.dumps({'counts': parameter_counts(config), 'config': config}, indent=2))
        return

    # 2. Each DP rank holds the full model and processes different samples.
    # Task gradients are averaged across ranks within each training step.
    training_device = initialize_device(device, cpu_threads)
    try:
        validate_config(config, world_size())

        # 3. Prepare the output directory and corpus. Resuming defaults to the
        # original run directory and data manifest.
        run_directory = resolve_output_directory(
            path_cfg, folder, output, resume, training_device,
        )
        # Calling main() directly uses the same local data as a no-argument CLI run.
        if (
            data_manifest is None
            and not config['data'].get('data_manifest')
            and not resume
            and not branch_from
            and path_cfg.resolve() == config_for_version('smoke_dclm_bf16').resolve()
        ):
            data_manifest = LOCAL_SMOKE_MANIFEST
        corpus = prepare_corpus(
            config, run_directory, training_device, data_manifest=data_manifest,
            resume=resume, branch_from=branch_from, allow_synthetic=allow_synthetic,
        )

        # 4. Trainer handles task optimization; its ConsensusManager handles
        # SALAAD structure. Set salaad.enabled=false for the vanilla MoE baseline.
        trainer = Trainer(config, corpus, training_device)
        return trainer.run(
            run_directory, resume=resume, stop_after=num_total_iters, branch_from=branch_from,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    # Keep assignments and main(...) explicit so launch arguments are easy to
    # inspect and edit in the IDE.
    args = parse_args()
    cfg_version = args.cfg_version
    folder = args.folder
    path_cfg = args.config or os.path.join(
        root, 'configs', cfg_version if cfg_version.endswith('.yaml') else cfg_version + '.yaml',
    )

    main(cfg_version, path_cfg, folder,
         args.num_total_iters,
         args.rho,
         data_manifest=args.data_manifest,
         output=args.output,
         resume=args.resume,
         branch_from=args.branch_from,
         device=args.device,
         cpu_threads=args.cpu_threads,
         allow_synthetic=args.allow_synthetic,
         dry_run=args.dry_run)
