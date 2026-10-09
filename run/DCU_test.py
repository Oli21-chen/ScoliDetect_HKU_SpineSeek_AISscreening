"""
DCU launcher for test/evaluation.

This script keeps the same testing pipeline as run_test.py,
but adds DCU-friendly visible-device diagnostics and optional env enforcement.

Typical usage:
  python DCU_test.py --gpu-ids 0,1
  python DCU_test.py --gpu-ids 0,1 --force-visible
  # Extra flags after launcher args are forwarded to run_test.py
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
    parser = argparse.ArgumentParser(description="Run test/evaluation on Hygon DCU")
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
    args, script_argv = parser.parse_known_args()

    requested_gpu_ids = _parse_gpu_ids(args.gpu_ids)
    print(f"[DCU Launcher][Test] requested_gpu_ids={requested_gpu_ids}")
    print("[DCU Launcher][Test] Runtime before env update:")
    pretty_print_runtime_devices(requested_gpu_ids=requested_gpu_ids)

    updated = apply_visible_devices(
        requested_gpu_ids,
        force=args.force_visible,
        update_cuda_visible=True,
    )
    print(f"[DCU Launcher][Test] env after apply_visible_devices(force={args.force_visible}): {updated}")
    print("[DCU Launcher][Test] Runtime after env update (before test entry):")
    pretty_print_runtime_devices(requested_gpu_ids=requested_gpu_ids)

    # Keep compatibility with existing testing entrypoint.
    target_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_test.py")
    sys.argv = [target_script, *script_argv]
    runpy.run_path(target_script, run_name="__main__")


if __name__ == "__main__":
    main()

