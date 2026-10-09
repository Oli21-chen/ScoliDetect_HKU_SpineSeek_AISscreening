"""
DCU launcher for training entrypoints (SFT k-fold or SigLIP pretrain).

Adds DCU-friendly visible-device diagnostics and optional env enforcement,
then runs the selected training script via runpy (config lives inside that script).

Typical usage:
  python DCU_train.py --gpu-ids 0,1
  python DCU_train.py --gpu-ids 0,1 --script run_end2end_kfold_vivit.py
  python DCU_train.py --gpu-ids 0,1 --script run_pretrain.py
  python DCU_train.py --gpu-ids 0,1 --script run_pretrain.py --condition peak
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from typing import List

from utils.dcu_utils import (
    apply_visible_devices,
    pretty_print_runtime_devices,
)


def _parse_gpu_ids(raw: str) -> List[int]:
    ids: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        ids.append(int(token))
    if len(ids) == 0:
        raise ValueError("gpu ids cannot be empty, e.g. --gpu-ids 0,1")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ViViT k-fold training on Hygon DCU")
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0,1",
        help="Comma-separated GPU ids to expose, e.g. 0,1",
    )
    parser.add_argument(
        "--force-visible",
        action="store_true",
        help="Force overwrite ROCR/HIP/CUDA_VISIBLE_DEVICES to --gpu-ids",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="run_end2end_kfold_vivit.py",
        help=(
            "Training entry script under the repo root, e.g. "
            "run_end2end_kfold_vivit.py or run_pretrain.py"
        ),
    )
    args, script_argv = parser.parse_known_args()

    requested_gpu_ids = _parse_gpu_ids(args.gpu_ids)
    print(f"[DCU Launcher] requested_gpu_ids={requested_gpu_ids}")
    print("[DCU Launcher] Runtime before env update:")
    pretty_print_runtime_devices(requested_gpu_ids=requested_gpu_ids)

    updated = apply_visible_devices(
        requested_gpu_ids,
        force=args.force_visible,
        update_cuda_visible=True,
    )
    print(f"[DCU Launcher] env after apply_visible_devices(force={args.force_visible}): {updated}")
    print("[DCU Launcher] Runtime after env update (before training entry):")
    pretty_print_runtime_devices(requested_gpu_ids=requested_gpu_ids)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.basename(args.script)
    target_script = os.path.join(repo_root, script_name)
    if not os.path.isfile(target_script):
        raise FileNotFoundError(
            f"Training script not found: {target_script}\n"
            f"Use --script with a file in {repo_root} (e.g. run_pretrain.py)."
        )
    print(f"[DCU Launcher] entry script: {target_script}")
    # Child scripts parse sys.argv; strip launcher flags so only their args remain.
    sys.argv = [target_script, *script_argv]
    runpy.run_path(target_script, run_name="__main__")


if __name__ == "__main__":
    main()
