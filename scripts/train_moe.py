"""MoE CLI alias; all training runs through main() in train_salad.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salaad_moe.config import config_for_version
from scripts.train_salad import main, parse_args


if __name__ == '__main__':
    args = parse_args()
    cfg_version = args.cfg_version
    folder = args.folder
    path_cfg = Path(args.config) if args.config else config_for_version(cfg_version)

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
