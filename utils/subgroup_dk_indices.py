"""
DK test cohort subgroup indices (e.g. ``data/subgroup_indices.json``).

Each JSON row's ``index`` is interpreted as a **patch identifier** that must map to
exactly one row of ``SigLIPFullGaitDatasetPKL.patch_metadata``:

- If ``patch_metadata`` is provided, ``index`` is resolved via ``int(patch_id)`` on
  each metadata entry (shuffle-safe). For the usual layout ``patch_00000009.pkl``,
  this is the number **9**, same as the 0-based dataset position when patches are
  consecutive.
- If ``patch_metadata`` is omitted (or present but no parseable ``patch_id``),
  ``index`` is treated as a 0-based dataset position (legacy).

The first clinical row may use ``index`` 9 (patches 0–8 absent from the subgroup
file); that is **not** an error. Warnings apply when a JSON ``index`` does not
match any loaded DK patch — for example the JSON lists patch ids **329–343**
but the DK folder only contains **329** files (e.g. ids **0–328**): those tail
rows have no file and are skipped. That is a cohort / export mismatch, not an
off-by-one from "starting at 9".
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple


def build_patch_id_to_position(patch_metadata: List[Dict[str, Any]]) -> Dict[int, int]:
    """Map numeric ``patch_id`` from metadata to 0-based dataset index (last wins on duplicates)."""
    out: Dict[int, int] = {}
    for i, entry in enumerate(patch_metadata):
        pid = entry.get("patch_id")
        if pid is None:
            continue
        try:
            k = int(str(pid).strip())
        except (TypeError, ValueError):
            continue
        out[k] = i
    return out


def resolve_subgroup_dataset_index(
    raw: int,
    n_patches: int,
    *,
    id_to_pos: Optional[Dict[int, int]] = None,
) -> Optional[int]:
    """
    Map JSON ``index`` to a dataset position in ``[0, n_patches)``, or None if unknown.

    If ``id_to_pos`` is non-empty, ``raw`` is treated **only** as a numeric
    ``patch_id`` (from ``patch_<id>.pkl`` / metadata). There is **no** fallback
    to legacy 0-based positions in that mode, so a missing id does not silently
    map to the wrong row.

    If ``id_to_pos`` is None or empty, ``raw`` is treated as a legacy 0-based
    dataset index when ``0 <= raw < n_patches``.
    """
    use_patch_ids = id_to_pos is not None and len(id_to_pos) > 0
    if use_patch_ids:
        pos = id_to_pos.get(raw)
        if pos is None:
            return None
        if 0 <= pos < n_patches:
            return pos
        return None
    if 0 <= raw < n_patches:
        return raw
    return None


def load_subgroup_dk_rows(path: str) -> List[Dict[str, Any]]:
    """Load raw subgroup rows (list of dicts with index, label, label1, label2, ...)."""
    with open(path, "rb") as f:
        text = f.read().decode("utf-8-sig")
    raw = json.loads(text)
    items = raw.get("subgroup", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise TypeError(f"Expected subgroup list in {path}, got {type(items)}")
    return list(items)


def load_dk_control_patch_ids(test_indices_path: str) -> set[int]:
    """
    Healthy-control patch ids from ``test_indices_dk.json`` (``max(label) == 0``).

    These are excluded from AIS interpretability; clinical AIS rows live in
    ``subgroup_indices.json`` with ``label1`` in {single, multi}.
    """
    with open(test_indices_path, "rb") as f:
        text = f.read().decode("utf-8-sig")
    raw = json.loads(text)
    items = raw.get("train", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise TypeError(f"Expected train list in {test_indices_path}, got {type(items)}")
    out: set[int] = set()
    for row in items:
        try:
            pid = int(row["index"])
        except (KeyError, TypeError, ValueError):
            continue
        mx = _max_cobb_from_label(row.get("label"))
        if mx is not None and mx == 0.0:
            out.add(pid)
    return out


def filter_strata_indices_ais_only(
    indices_by_name: Dict[str, List[int]],
    *,
    full_dataset,
    n_dk: int,
    binary_threshold: float,
    control_patch_ids: Optional[set[int]] = None,
    exclude_screening_negative: bool = True,
) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    """
    Restrict interpretability strata to DK AIS patches only.

    - Drops non-DK indices when ``test_dataset`` is a DK+PK ``ConcatDataset``.
    - Drops healthy controls (``control_patch_ids`` from ``test_indices_dk.json``).
    - Optionally drops screening-negative patches (binary label 0 at ``binary_threshold``).
    """
    from utils.utils import _get_label_for_index

    control_patch_ids = control_patch_ids or set()
    patch_metadata = getattr(full_dataset, "patch_metadata", None)
    if patch_metadata is None and hasattr(full_dataset, "datasets"):
        for ds in full_dataset.datasets:
            if hasattr(ds, "patch_metadata"):
                patch_metadata = ds.patch_metadata
                break
    id_to_pos = build_patch_id_to_position(patch_metadata) if patch_metadata else {}
    pos_to_pid = {pos: pid for pid, pos in id_to_pos.items()}

    stats: Dict[str, Any] = {
        "n_dk": int(n_dk),
        "binary_threshold": float(binary_threshold),
        "exclude_screening_negative": bool(exclude_screening_negative),
        "n_control_patch_ids": int(len(control_patch_ids)),
        "per_stratum": {},
    }

    def _keep(idx: int) -> bool:
        if idx < 0 or idx >= n_dk:
            return False
        pid = pos_to_pid.get(idx)
        if pid is not None and pid in control_patch_ids:
            return False
        if exclude_screening_negative:
            if _get_label_for_index(full_dataset, idx, binary_threshold) != 1:
                return False
        return True

    out: Dict[str, List[int]] = {}
    for name, idx_list in indices_by_name.items():
        before = len(idx_list)
        kept = [i for i in idx_list if _keep(i)]
        out[name] = kept
        stats["per_stratum"][name] = {"before": before, "after": len(kept)}
    stats["ais_union_n"] = len(set().union(*out.values())) if out else 0
    return out, stats


def _max_cobb_from_label(label_val: Any) -> Optional[float]:
    if label_val is None:
        return None
    if isinstance(label_val, (list, tuple)):
        try:
            return float(max(float(x) for x in label_val))
        except (TypeError, ValueError):
            return None
    try:
        return float(label_val)
    except (TypeError, ValueError):
        return None


def warn_oob_subgroup_rows(
    rows: List[Dict[str, Any]],
    n_patches: int,
    patch_metadata: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Print a warning for JSON ``index`` values that do not resolve to any DK patch."""
    if patch_metadata is not None:
        id_to_pos = build_patch_id_to_position(patch_metadata)
    else:
        id_to_pos = None

    if patch_metadata is not None and len(patch_metadata) > 0 and len(id_to_pos) == 0:
        print(
            "Warning: `patch_metadata` is non-empty but no parseable `patch_id` fields "
            "were found. JSON `index` values are interpreted as 0-based dataset "
            "positions (legacy). Regenerate metadata or fix `patch_id` if indices "
            "should match `patch_<id>.pkl` stems."
        )

    oob_raw: List[int] = []
    all_raw: List[int] = []
    for row in rows:
        try:
            raw = int(row["index"])
        except (KeyError, TypeError, ValueError):
            continue
        all_raw.append(raw)
        if resolve_subgroup_dataset_index(raw, n_patches, id_to_pos=id_to_pos) is None:
            oob_raw.append(raw)
    if not oob_raw:
        return
    oob_raw = sorted(set(oob_raw))
    max_valid = n_patches - 1
    extra = ""
    if id_to_pos is not None and len(id_to_pos) > 0:
        pid_min = min(id_to_pos.keys())
        pid_max = max(id_to_pos.keys())
        j_min = min(all_raw) if all_raw else None
        j_max = max(all_raw) if all_raw else None
        extra = (
            f"Resolution mode: JSON `index` = numeric patch_id (shuffle-safe). "
            f"DK patch_id range in metadata: {pid_min}..{pid_max} ({len(id_to_pos)} files). "
            f"JSON index range in file: {j_min}..{j_max}. "
        )
    else:
        extra = (
            f"Resolution mode: legacy 0-based dataset index in 0..{max_valid}. "
        )
    print(
        f"Warning: {len(oob_raw)} distinct JSON `index` values do not match any DK patch "
        f"(dataset len={n_patches}). They are skipped. {extra}"
        f"Example missing values: {oob_raw[:8]}{'...' if len(oob_raw) > 8 else ''}"
    )


def build_dk_strata_indices(
    subgroup_rows: List[Dict[str, Any]],
    n_patches: int,
    *,
    patch_metadata: Optional[List[Dict[str, Any]]] = None,
    cobb_general_threshold: float = 10.0,
    cobb_general_strict_gt: bool = True,
    control_patch_ids: Optional[set[int]] = None,
    ais_only: bool = False,
) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    """
    Build stratum index lists from JSON rows (in-range indices only).

    Strata:
        - ``general_cobb_gt10``: ``max(label) > threshold`` (or ``>=`` if strict_gt=False)
        - ``single``: ``label1`` single (any ``label2``; matches legacy chart1 "single")
        - ``multi``: ``label1`` multi
        - ``single_thoracic``: single + thoracic
        - ``single_lumbar``: single + lumbar

    Returns:
        (indices_by_name, meta) where meta includes counts and threshold flags.
    """
    general: set[int] = set()
    single_all: set[int] = set()
    multi: set[int] = set()
    single_th: set[int] = set()
    single_lb: set[int] = set()

    id_to_pos = build_patch_id_to_position(patch_metadata) if patch_metadata is not None else None
    control_patch_ids = control_patch_ids or set()

    for row in subgroup_rows:
        try:
            raw = int(row["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if raw in control_patch_ids:
            continue
        idx = resolve_subgroup_dataset_index(raw, n_patches, id_to_pos=id_to_pos)
        if idx is None:
            continue

        m1 = str(row.get("label1") or "").lower()
        m2_raw = row.get("label2")
        m2s = str(m2_raw).lower() if m2_raw is not None else ""

        if ais_only and m1 not in ("single", "multi"):
            continue

        mx = _max_cobb_from_label(row.get("label"))
        if mx is None:
            continue
        if cobb_general_strict_gt:
            cobb_ok = mx > cobb_general_threshold
        else:
            cobb_ok = mx >= cobb_general_threshold
        if not cobb_ok:
            continue

        general.add(idx)

        if m1 == "single":
            single_all.add(idx)
            if m2s == "thoracic":
                single_th.add(idx)
            elif m2s == "lumbar":
                single_lb.add(idx)
        elif m1 == "multi":
            multi.add(idx)

    def _sort(s: set[int]) -> List[int]:
        return sorted(s)

    indices = {
        "general_cobb_gt10": _sort(general),
        "single": _sort(single_all),
        "multi": _sort(multi),
        "single_thoracic": _sort(single_th),
        "single_lumbar": _sort(single_lb),
    }
    meta = {
        "n_patches": int(n_patches),
        "cobb_general_threshold": float(cobb_general_threshold),
        "cobb_general_strict_gt": bool(cobb_general_strict_gt),
        "ais_only": bool(ais_only),
        "n_control_patch_ids_excluded": int(len(control_patch_ids)),
        "counts": {k: len(v) for k, v in indices.items()},
    }
    return indices, meta


def subgroup_map_from_rows(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Collapse rows to a dict keyed by index (last row wins if duplicate keys).
    Values contain label1/label2 only (legacy shape for callers that need a map).
    """
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        try:
            idx = int(row["index"])
        except (KeyError, TypeError, ValueError):
            continue
        out[idx] = {
            "label1": row.get("label1"),
            "label2": row.get("label2"),
        }
    return out
