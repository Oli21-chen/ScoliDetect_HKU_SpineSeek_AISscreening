"""
Compare kinematic knowledge map (KM) distributions between pooled cohorts.
KM tensors are (T, F), typically (96, 238) per patch.

Default cohorts (override with --internal / --external; each accepts one or more dirs):
  Internal (pooled):  dataset/train_sz_pkl^1, dataset/test_sz_pkl^1  (n = 820)
  External (pooled):  dataset/test_pk_pkl^1 (n = 155)
                      + dataset/test_dk_pkl^1 filtered to general Cobb > 10°
                        via dataset/subgroup_indices.json (n = 299) → total n = 454

Useful when validation/generalization metrics differ across sites: cohort-specific
pipelines shift where signal lives in (time, feature) space and how dispersed
patches are, independent of model AUC.

Outputs (default: tools/km_distribution_plots/; Nature-style 300 dpi PNG):
  - km_mean_heatmaps.png          : mean KM side-by-side
  - km_std_heatmaps.png           : per-cell std across patches (within-cohort spread)
  - km_mean_diff.png              : mean(internal) - mean(external)
  - km_patch_norm_hist.png        : Frobenius norm per patch (global "energy")
  - km_temporal_sum_curve.png     : sum over F vs time (mean ± s.d.)
  - km_pca_scatter.png            : joint PCA (PC axes show % explained variance)
  - km_distribution_metrics.json  : cohort scalars, per-directory patch counts, PCA, std tails

Run from project root:
  python tools/km_distribution_visualization.py
  python tools/km_distribution_visualization.py --internal ./dataset/train_sz_pkl^1 ./dataset/test_sz_pkl^1 \\
      --external ./dataset/test_pk_pkl^1 ./dataset/test_dk_pkl^1

Figure style follows common Nature-family conventions: sans-serif fonts (~7–8 pt
equivalent at 300 dpi), minimal spines, colorblind-friendly cohort colours, and
double/single-column widths (~180 mm / ~87 mm).
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path
from typing import AbstractSet, Dict, List, Optional, Set, Tuple

import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --- Nature-style figure setup (see Nature "Guide to authors" / figure preparation) ---
# Widths in inches: double column ~180 mm; single column ~87 mm.
DOUBLE_COL_W = 7.087  # 180 mm
SINGLE_COL_W = 3.425  # 87 mm

# Okabe–Ito colorblind-safe palette (approx. internal / external)
COLOR_INTERNAL = "#0072B2"
COLOR_EXTERNAL = "#E69F00"

_NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def _style_axis(ax: mpl.axes.Axes) -> None:
    ax.tick_params(axis="both", which="major", labelsize=7, length=3, width=0.6)
    ax.tick_params(axis="both", which="minor", length=2, width=0.4)


def _panel_label(ax: mpl.axes.Axes, letter: str, x: float = -0.12, y: float = 1.02) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def _save_figure(fig: mpl.figure.Figure, out_dir: str, basename: str, formats: List[str]) -> None:
    """Write figure at publication resolution (default: 300 dpi PNG; optional formats via --formats)."""
    for fmt in formats:
        fmt = fmt.lower().strip().lstrip(".")
        path = os.path.join(out_dir, f"{basename}.{fmt}")
        save_kw: Dict[str, object] = {
            "format": fmt,
            "bbox_inches": "tight",
            "pad_inches": 0.02,
        }
        if fmt == "png":
            save_kw["dpi"] = 300
        fig.savefig(path, **save_kw)


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def _install_numpy_pickle_compat() -> None:
    """
    Map ``numpy._core.*`` when pickles were saved with NumPy 2.x but loaded on NumPy 1.x.
    """
    if hasattr(np, "_core"):
        return
    import numpy.core as _core

    sys.modules.setdefault("numpy._core", _core)
    for sub in (
        "numeric",
        "multiarray",
        "_multiarray_umath",
        "umath",
        "fromnumeric",
        "_dtype",
    ):
        modname = f"numpy._core.{sub}"
        if modname not in sys.modules:
            try:
                sys.modules[modname] = getattr(_core, sub)
            except AttributeError:
                pass


def _pickle_load(file_obj) -> object:
    _install_numpy_pickle_compat()
    return pickle.load(file_obj)


_install_numpy_pickle_compat()


def _patch_metadata_length(pkl_data_dir_abs: str) -> int:
    metadata_path = os.path.join(pkl_data_dir_abs, "patch_metadata.pkl")
    if os.path.isfile(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = _pickle_load(f)
        return len(metadata["patch_metadata"])
    patches_dir = os.path.join(pkl_data_dir_abs, "patches")
    return len(glob.glob(os.path.join(patches_dir, "*.pkl")))


def _load_subgroup_rows(subgroup_json: str) -> List[dict]:
    with open(subgroup_json, "rb") as f:
        text = f.read().decode("utf-8-sig")
    raw = json.loads(text)
    items = raw.get("subgroup", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise TypeError(f"Expected subgroup list in {subgroup_json}")
    return list(items)


def _dk_general_cobb_gt10_indices(
    subgroup_json: str,
    n_patches: int,
    *,
    cobb_threshold: float = 10.0,
) -> Set[int]:
    """
    Patch indices for DK external eval (Dataset 1): annotated rows with max Cobb > threshold.
    Matches ``build_dk_strata_indices`` → ``general_cobb_gt10`` (expected n = 299).
    """
    allowed: Set[int] = set()
    for row in _load_subgroup_rows(subgroup_json):
        try:
            idx = int(row["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if idx < 0 or idx >= n_patches:
            continue
        label = row.get("label")
        if not isinstance(label, (list, tuple)) or len(label) < 2:
            continue
        mx = max(float(label[0]), float(label[1]))
        if mx > cobb_threshold:
            allowed.add(idx)
    return allowed


def _resolve_pkl_path(entry: dict, pkl_data_dir_abs: str) -> str:
    raw = entry.get("pkl_path", "")
    if not os.path.isabs(raw):
        path = os.path.normpath(os.path.abspath(raw))
    else:
        path = os.path.normpath(raw)
    if not os.path.exists(path):
        pid = entry.get("patch_id")
        if pid is None:
            pid = Path(entry.get("pkl_path", "")).stem.replace("patch_", "")
        fallback = os.path.join(pkl_data_dir_abs, "patches", f"patch_{pid}.pkl")
        if os.path.exists(fallback):
            path = fallback
    return path


def load_knowledge_map_stack(
    pkl_data_dir_abs: str,
    name: str,
    *,
    allowed_patch_indices: Optional[AbstractSet[int]] = None,
) -> np.ndarray:
    """
    Load patches under a preprocessed PKL directory; return (N, T, F).

    If ``allowed_patch_indices`` is set, only rows whose enumerate index is in the set
    are loaded (aligns with ``subgroup_indices.json`` ``index`` field).
    """
    metadata_path = os.path.join(pkl_data_dir_abs, "patch_metadata.pkl")
    if not os.path.exists(metadata_path):
        patches_dir = os.path.join(pkl_data_dir_abs, "patches")
        patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.pkl")))
        if not patch_files:
            raise FileNotFoundError(f"No metadata and no patch files in {patches_dir}")
        patch_metadata = []
        for pkl_path in patch_files:
            try:
                with open(pkl_path, "rb") as f:
                    _pickle_load(f)
                patch_id = Path(pkl_path).stem.replace("patch_", "")
                patch_metadata.append({"patch_id": patch_id, "pkl_path": pkl_path})
            except Exception as e:
                print(f"Warning [{name}]: skip {pkl_path}: {e}")
                continue
    else:
        with open(metadata_path, "rb") as f:
            metadata = _pickle_load(f)
        patch_metadata = metadata["patch_metadata"]

    km_list: List[np.ndarray] = []
    for patch_idx, entry in enumerate(patch_metadata):
        if allowed_patch_indices is not None and patch_idx not in allowed_patch_indices:
            continue
        pkl_path = _resolve_pkl_path(entry, pkl_data_dir_abs)
        if not os.path.exists(pkl_path):
            continue
        try:
            with open(pkl_path, "rb") as f:
                patch_data = _pickle_load(f)
            km = patch_data["knowledge_map"]
            if hasattr(km, "numpy"):
                km = km.numpy()
            arr = np.asarray(km, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(f"expected 2D knowledge_map, got shape {arr.shape}")
            km_list.append(arr)
        except Exception as e:
            print(f"Warning [{name}]: skip {pkl_path}: {e}")
            continue

    if not km_list:
        raise ValueError(f"No knowledge_map loaded for {name}")
    return np.stack(km_list, axis=0)


def load_pooled_knowledge_maps(
    directories: List[str],
    cohort_tag: str,
    *,
    directory_patch_filters: Optional[Dict[str, AbstractSet[int]]] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Load KM from several preprocessed PKL roots and concatenate along the patch axis.
    Returns (km_all, patch_counts_by_dir_basename).

    ``directory_patch_filters`` maps directory basename → allowed patch indices (optional).
    """
    if not directories:
        raise ValueError(f"{cohort_tag}: at least one directory is required")
    chunks: List[np.ndarray] = []
    counts: Dict[str, int] = {}
    tf_shape: Optional[Tuple[int, int]] = None

    for raw in directories:
        d_abs = os.path.abspath(os.path.normpath(raw))
        if not os.path.isdir(d_abs):
            raise FileNotFoundError(f"{cohort_tag} directory not found: {d_abs}")
        key = os.path.basename(d_abs.rstrip(os.sep))
        allowed = None
        if directory_patch_filters:
            allowed = directory_patch_filters.get(key)
        arr = load_knowledge_map_stack(d_abs, key, allowed_patch_indices=allowed)
        counts[key] = int(arr.shape[0])
        if allowed is not None:
            print(
                f"  [{cohort_tag}] {key}: filter kept {counts[key]}/{len(allowed)} "
                f"indexed patches (Cobb > threshold subset)"
            )
        if tf_shape is None:
            tf_shape = (int(arr.shape[1]), int(arr.shape[2]))
        elif (int(arr.shape[1]), int(arr.shape[2])) != tf_shape:
            raise ValueError(
                f"{cohort_tag}: KM shape {arr.shape} in {d_abs} does not match "
                f"(*, {tf_shape[0]}, {tf_shape[1]}) from earlier directories"
            )
        chunks.append(arr)
        print(f"  [{cohort_tag}] {key}: {arr.shape[0]} patches, KM shape {tuple(arr.shape[1:])}")

    pooled = np.concatenate(chunks, axis=0)
    print(f"  [{cohort_tag}] pooled: {pooled.shape[0]} patches total")
    return pooled, counts


def _cohort_metrics(km: np.ndarray) -> Dict[str, float]:
    """Scalar summaries for console / figure captions."""
    n = km.shape[0]
    flat = km.reshape(n, -1)
    norms = np.linalg.norm(flat, axis=1)
    per_cell_std = np.std(km, axis=0)
    return {
        "n": float(n),
        "mean_frobenius": float(np.mean(norms)),
        "std_frobenius": float(np.std(norms)),
        "mean_per_cell_std": float(np.mean(per_cell_std)),
        "median_per_cell_std": float(np.median(per_cell_std)),
    }


def _pca2_joint(
    flat_a: np.ndarray, flat_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    PC1–PC2 from rows of [A; B], globally centered (shared basis for both cohorts).
    Returns (z_a, z_b, singular_values, total_var) where total_var = sum of squared
    centered values (for explained-variance ratios of PC1/PC2).
    """
    x = np.vstack([flat_a, flat_b]).astype(np.float64, copy=False)
    mu = np.mean(x, axis=0, keepdims=True)
    xc = x - mu
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    total_var = float(np.sum(xc ** 2))
    za = (flat_a.astype(np.float64, copy=False) - mu) @ vt[:2].T
    zb = (flat_b.astype(np.float64, copy=False) - mu) @ vt[:2].T
    return za, zb, s, total_var


def plot_comparisons(
    internal: np.ndarray,
    external: np.ndarray,
    labels: Tuple[str, str],
    out_dir: str,
    std_vmax_percentile: Optional[float] = None,
    save_formats: Optional[List[str]] = None,
) -> Dict[str, object]:
    formats = save_formats if save_formats else ["png"]
    os.makedirs(out_dir, exist_ok=True)
    li, le = labels

    mi = np.mean(internal, axis=0)
    me = np.mean(external, axis=0)
    si = np.std(internal, axis=0)
    se = np.std(external, axis=0)

    t_bins, f_bins = mi.shape
    extent = [0, f_bins, t_bins, 0]

    std_vmax: Optional[float] = None
    if std_vmax_percentile is not None:
        pct = float(std_vmax_percentile)
        std_vmax = float(
            max(
                np.percentile(si, pct),
                np.percentile(se, pct),
            )
        )

    def _colorbar_nature(im, ax: mpl.axes.Axes) -> None:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.025, shrink=0.92)
        cbar.ax.tick_params(labelsize=7, width=0.5, length=2.5)
        cbar.outline.set_linewidth(0.5)

    def _heatmap(
        ax,
        data,
        title,
        cmap="viridis",
        center_cmap=False,
        vmin=None,
        vmax=None,
    ):
        if center_cmap:
            lim = np.nanmax(np.abs(data)) or 1.0
            vmin, vmax = -lim, lim
        im = ax.imshow(
            data,
            aspect="auto",
            cmap=cmap,
            extent=extent,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Time index")
        ax.set_title(title, pad=6)
        _style_axis(ax)
        _colorbar_nature(im, ax)
        return im

    with plt.rc_context(_NATURE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 2.95))
        # _heatmap(axes[0], mi, f"Mean KM, internal (n = {internal.shape[0]})")
        _heatmap(axes[0], mi, f"Mean KM, internal (n = {internal.shape[0]})")
        _heatmap(axes[1], me, f"Mean KM, external (n = {external.shape[0]})")
        _panel_label(axes[0], "a")
        _panel_label(axes[1], "b")
        fig.tight_layout(w_pad=1.2)
        _save_figure(fig, out_dir, "km_mean_heatmaps", formats)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 2.95))
        _heatmap(
            axes[0],
            si,
            "Std (across patches), internal",
            vmax=std_vmax,
            vmin=0.0,
        )
        _heatmap(
            axes[1],
            se,
            "Std (across patches), external",
            vmax=std_vmax,
            vmin=0.0,
        )
        _panel_label(axes[0], "a")
        _panel_label(axes[1], "b")
        if std_vmax is not None:
            fig.suptitle(
                f"Shared colour scale cap: {std_vmax_percentile:.0f}th percentile (both cohorts)",
                fontsize=8,
                y=1.01,
                fontweight="normal",
            )
        fig.tight_layout(w_pad=1.2)
        _save_figure(fig, out_dir, "km_std_heatmaps", formats)
        plt.close(fig)

        diff = mi - me
        vmax = np.nanmax(np.abs(diff)) or 1.0
        fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 2.75))
        im = ax.imshow(
            diff,
            aspect="auto",
            cmap="RdBu_r",
            extent=extent,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Time index")
        ax.set_title("Mean KM difference (internal − external)", pad=6)
        _style_axis(ax)
        _colorbar_nature(im, ax)
        _panel_label(ax, "a", x=-0.06, y=1.02)
        fig.tight_layout()
        _save_figure(fig, out_dir, "km_mean_diff", formats)
        plt.close(fig)

        ni = np.linalg.norm(internal.reshape(internal.shape[0], -1), axis=1)
        ne = np.linalg.norm(external.reshape(external.shape[0], -1), axis=1)
        fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.45))
        lo = float(min(ni.min(), ne.min()))
        hi = float(max(ni.max(), ne.max()))
        bins = np.linspace(lo, hi, 38)
        ax.hist(
            ni,
            bins=bins,
            alpha=0.65,
            label=f"Internal (n = {len(ni)})",
            density=True,
            color=COLOR_INTERNAL,
            histtype="stepfilled",
            edgecolor=COLOR_INTERNAL,
            linewidth=0.25,
        )
        ax.hist(
            ne,
            bins=bins,
            alpha=0.65,
            label=f"External (n = {len(ne)})",
            density=True,
            color=COLOR_EXTERNAL,
            histtype="stepfilled",
            edgecolor=COLOR_EXTERNAL,
            linewidth=0.25,
        )
        ax.set_xlabel(r"$\Vert\mathrm{KM}\Vert_\mathrm{F}$ per patch")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")
        _style_axis(ax)
        ax.grid(True, alpha=0.18, linewidth=0.4, linestyle="-")
        _panel_label(ax, "a", x=-0.18, y=1.02)
        fig.tight_layout()
        _save_figure(fig, out_dir, "km_patch_norm_hist", formats)
        plt.close(fig)

        xi = np.arange(t_bins)
        sum_i = internal.sum(axis=-1)
        sum_e = external.sum(axis=-1)
        mean_i, std_i = np.mean(sum_i, axis=0), np.std(sum_i, axis=0)
        mean_e, std_e = np.mean(sum_e, axis=0), np.std(sum_e, axis=0)
        fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 2.45))
        ax.fill_between(
            xi,
            mean_i - std_i,
            mean_i + std_i,
            alpha=0.22,
            color=COLOR_INTERNAL,
            linewidth=0,
        )
        ax.plot(xi, mean_i, label="Internal", color=COLOR_INTERNAL, linewidth=1.2)
        ax.fill_between(
            xi,
            mean_e - std_e,
            mean_e + std_e,
            alpha=0.22,
            color=COLOR_EXTERNAL,
            linewidth=0,
        )
        ax.plot(xi, mean_e, label="External", color=COLOR_EXTERNAL, linewidth=1.2)
        ax.set_xlabel("Time index")
        ax.set_ylabel(r"$\sum_f$ KM$(t,f)$")
        ax.set_title("Temporal profile (mean ± s.d.)", pad=6)
        ax.legend(loc="best")
        _style_axis(ax)
        ax.grid(True, alpha=0.18, linewidth=0.4, linestyle="-")
        _panel_label(ax, "a", x=-0.06, y=1.02)
        fig.tight_layout()
        _save_figure(fig, out_dir, "km_temporal_sum_curve", formats)
        plt.close(fig)

        fi = internal.reshape(internal.shape[0], -1)
        fe = external.reshape(external.shape[0], -1)
        zi, ze, s_svd, total_var = _pca2_joint(fi, fe)
        ev1 = float((s_svd[0] ** 2) / total_var) if total_var > 0 else 0.0
        ev2 = float((s_svd[1] ** 2) / total_var) if total_var > 0 else 0.0
        fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.85))
        ax.scatter(
            zi[:, 0],
            zi[:, 1],
            s=10,
            alpha=0.45,
            label="Internal",
            c=COLOR_INTERNAL,
            edgecolors="none",
            rasterized=True,
        )
        ax.scatter(
            ze[:, 0],
            ze[:, 1],
            s=10,
            alpha=0.45,
            label="External",
            c=COLOR_EXTERNAL,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_xlabel(f"PC1 ({ev1 * 100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({ev2 * 100:.1f}% variance)")
        ax.set_title("PCA of vectorized KM (pooled basis)", pad=6)
        ax.legend(loc="best", markerscale=1.2)
        _style_axis(ax)
        ax.grid(True, alpha=0.18, linewidth=0.4, linestyle="-")
        _panel_label(ax, "a", x=-0.18, y=1.02)
        fig.tight_layout()
        _save_figure(fig, out_dir, "km_pca_scatter", formats)
        plt.close(fig)

    metrics: Dict[str, object] = {
        "internal": _cohort_metrics(internal),
        "external": _cohort_metrics(external),
        "km_shape": [int(internal.shape[1]), int(internal.shape[2])],
        "pca_joint": {
            "explained_variance_ratio_pc1": ev1,
            "explained_variance_ratio_pc2": ev2,
            "explained_variance_ratio_pc1_pc2": ev1 + ev2,
        },
        "per_cell_std_max": {
            "internal": float(np.nanmax(si)),
            "external": float(np.nanmax(se)),
        },
        "per_cell_std_p99": {
            "internal": float(np.percentile(si, 99)),
            "external": float(np.percentile(se, 99)),
        },
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize KM distribution: pooled internal vs pooled external directories."
    )
    parser.add_argument(
        "--internal",
        nargs="+",
        default=[
            os.path.join(parent_dir, "dataset", "train_sz_pkl^1"),
            os.path.join(parent_dir, "dataset", "test_sz_pkl^1"),
        ],
        help="One or more PKL roots for the internal (e.g. SZ) cohort; patches are concatenated.",
    )
    parser.add_argument(
        "--external",
        nargs="+",
        default=[
            os.path.join(parent_dir, "dataset", "test_pk_pkl^1"),
            os.path.join(parent_dir, "dataset", "test_dk_pkl^1"),
        ],
        help="One or more PKL roots for the external cohort; patches are concatenated.",
    )
    parser.add_argument(
        "--internal-label",
        default="Internal (SZ, train+test pooled)",
        help="Legend / title label for the internal cohort",
    )
    parser.add_argument(
        "--external-label",
        default="External (PK + DK eval, n=454)",
        help="Legend / title label for the external cohort",
    )
    parser.add_argument(
        "--external-dk-subgroup-json",
        type=Path,
        default=os.path.join(parent_dir, "dataset", "subgroup_indices.json"),
        help="Subgroup JSON for DK patch filter (general Cobb > threshold).",
    )
    parser.add_argument(
        "--external-dk-cobb-threshold",
        type=float,
        default=10.0,
        help="Keep DK patches with max(M_cobb_l, M_cobb_r) > this value (default 10°).",
    )
    parser.add_argument(
        "--external-dk-all-patches",
        action="store_true",
        help="Use all DK PKL patches (329) instead of Cobb-filtered eval subset (299).",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(current_dir, "km_distribution_plots"),
        help="Directory for PNG outputs",
    )
    parser.add_argument(
        "--std-vmax-percentile",
        type=float,
        default=None,
        metavar="P",
        help="If set (e.g. 99), cap std heatmap colors at max of P-th percentile "
        "across both cohorts so panels are directly comparable (raw max still in JSON).",
    )
    parser.add_argument(
        "--formats",
        default="png",
        help="Comma-separated outputs, e.g. png or png,pdf,svg (300 dpi for PNG).",
    )
    args = parser.parse_args()

    print("Loading internal cohort (pooled):")
    km_i, internal_counts = load_pooled_knowledge_maps(list(args.internal), "internal")

    external_filters: Optional[Dict[str, AbstractSet[int]]] = None
    if not args.external_dk_all_patches:
        external_filters = {}
        for raw in args.external:
            d_abs = os.path.abspath(os.path.normpath(raw))
            key = os.path.basename(d_abs.rstrip(os.sep))
            if "test_dk_pkl" not in key:
                continue
            n_dk = _patch_metadata_length(d_abs)
            dk_allowed = _dk_general_cobb_gt10_indices(
                str(args.external_dk_subgroup_json),
                n_dk,
                cobb_threshold=args.external_dk_cobb_threshold,
            )
            if len(dk_allowed) == 0:
                raise RuntimeError(
                    f"No DK patches passed Cobb > {args.external_dk_cobb_threshold}° filter; "
                    f"check {args.external_dk_subgroup_json}"
                )
            external_filters[key] = dk_allowed
            print(
                f"External DK filter ({key}): max Cobb > {args.external_dk_cobb_threshold}° "
                f"→ {len(dk_allowed)} / {n_dk} patches"
            )
        if not external_filters:
            external_filters = None

    print("Loading external cohort (pooled):")
    km_e, external_counts = load_pooled_knowledge_maps(
        list(args.external),
        "external",
        directory_patch_filters=external_filters,
    )

    if km_i.shape[1:] != km_e.shape[1:]:
        raise ValueError(
            f"KM shape mismatch: internal {km_i.shape} vs external {km_e.shape}"
        )

    mi = _cohort_metrics(km_i)
    me = _cohort_metrics(km_e)
    print("\nCohort summaries (Frobenius norm per patch; per-cell std across patches):")
    for k, v in mi.items():
        print(f"  internal  {k}: {v}")
    for k, v in me.items():
        print(f"  external  {k}: {v}")

    labels = (args.internal_label, args.external_label)
    save_formats = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    metrics = plot_comparisons(
        km_i,
        km_e,
        labels,
        args.out_dir,
        std_vmax_percentile=args.std_vmax_percentile,
        save_formats=save_formats,
    )
    metrics["internal_patches_by_directory"] = internal_counts
    metrics["external_patches_by_directory"] = external_counts
    metrics_path = os.path.join(args.out_dir, "km_distribution_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")
    if "pca_joint" in metrics:
        pj = metrics["pca_joint"]
        print(
            f"Joint PCA explained variance: PC1={pj['explained_variance_ratio_pc1']:.4f}, "
            f"PC2={pj['explained_variance_ratio_pc2']:.4f}"
        )
    print(f"Saved figures under: {args.out_dir}")


if __name__ == "__main__":
    main()
