"""Generate 238-feature index table for utili_Getpose_v4.py (one-off helper)."""
from __future__ import annotations

LANDMARKS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COORD_SUFFIX = ("_x", "_y")


def _avg(a: str, b: str) -> str:
    return f"avg({a},{b})"


def _dist_entries() -> list[tuple[str, str]]:
    """Each entry expands to 2 features (dx, dy)."""
    specs: list[tuple[str, str]] = []
    specs.append(
        (_avg("left_hip", "right_hip"), _avg("left_shoulder", "right_shoulder"))
    )
    name_pairs = [
        ("nose", "left_shoulder"),
        ("nose", "right_shoulder"),
        ("nose", "left_hip"),
        ("nose", "right_hip"),
        ("left_elbow", "left_shoulder"),
        ("left_elbow", "left_hip"),
        ("left_elbow", "left_knee"),
        ("left_elbow", "left_ankle"),
        ("left_elbow", "right_shoulder"),
        ("left_elbow", "right_hip"),
        ("left_elbow", "right_knee"),
        ("left_elbow", "right_ankle"),
        ("right_elbow", "left_shoulder"),
        ("right_elbow", "left_hip"),
        ("right_elbow", "left_knee"),
        ("right_elbow", "left_ankle"),
        ("right_elbow", "right_shoulder"),
        ("right_elbow", "right_hip"),
        ("right_elbow", "right_knee"),
        ("right_elbow", "right_ankle"),
        ("left_shoulder", "left_hip"),
        ("left_shoulder", "left_knee"),
        ("left_shoulder", "left_ankle"),
        ("left_shoulder", "right_hip"),
        ("left_shoulder", "right_knee"),
        ("left_shoulder", "right_ankle"),
        ("right_shoulder", "left_hip"),
        ("right_shoulder", "left_knee"),
        ("right_shoulder", "left_ankle"),
        ("right_shoulder", "right_hip"),
        ("right_shoulder", "right_knee"),
        ("right_shoulder", "right_ankle"),
        ("left_hip", "left_knee"),
        ("left_hip", "left_ankle"),
        ("left_hip", "right_knee"),
        ("left_hip", "right_ankle"),
        ("right_hip", "left_knee"),
        ("right_hip", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_hip", "right_ankle"),
        ("left_wrist", "right_wrist"),
        ("left_elbow", "right_elbow"),
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
    ]
    specs.extend(name_pairs)
    specs.extend(
        [
            (_avg("left_hip", "right_hip"), _avg("right_ankle", "right_ankle")),
            (_avg("left_hip", "right_hip"), _avg("left_ankle", "left_ankle")),
            (_avg("left_hip", "right_hip"), _avg("right_shoulder", "right_shoulder")),
            (_avg("left_hip", "right_hip"), _avg("left_shoulder", "left_shoulder")),
            (_avg("left_eye", "right_eye"), _avg("right_shoulder", "right_shoulder")),
            (_avg("left_eye", "right_eye"), _avg("left_shoulder", "left_shoulder")),
        ]
    )
    return specs


def _ang_entries() -> list[tuple[tuple[str, str], tuple[str, str]]]:
    # (v1_from, v1_to), (v2_from, v2_to) per _get_angle_by_names call order
    def a(v1, v2):
        return (tuple(v1), tuple(v2))

    return [
        a(["left_shoulder", "right_shoulder"], ["left_hip", "right_hip"]),
        a(["left_elbow", "left_shoulder"], ["right_elbow", "right_shoulder"]),
        a(["left_elbow", "left_shoulder"], ["right_hip", "right_shoulder"]),
        a(["left_elbow", "left_shoulder"], ["right_knee", "right_hip"]),
        a(["left_elbow", "left_shoulder"], ["right_ankle", "right_knee"]),
        a(["left_elbow", "left_shoulder"], ["left_hip", "left_shoulder"]),
        a(["left_elbow", "left_shoulder"], ["left_knee", "left_hip"]),
        a(["left_elbow", "left_shoulder"], ["left_ankle", "left_knee"]),
        a(["right_elbow", "right_shoulder"], ["right_hip", "right_shoulder"]),
        a(["right_elbow", "right_shoulder"], ["right_knee", "right_hip"]),
        a(["right_elbow", "right_shoulder"], ["right_ankle", "right_knee"]),
        a(["right_elbow", "right_shoulder"], ["left_hip", "left_shoulder"]),
        a(["right_elbow", "right_shoulder"], ["left_knee", "left_hip"]),
        a(["right_elbow", "right_shoulder"], ["left_ankle", "left_knee"]),
        a(["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"]),
        a(["left_shoulder", "left_hip"], ["right_knee", "right_hip"]),
        a(["left_shoulder", "left_hip"], ["right_ankle", "right_knee"]),
        a(["left_shoulder", "left_hip"], ["left_knee", "left_hip"]),
        a(["left_shoulder", "left_hip"], ["left_ankle", "left_knee"]),
        a(["right_shoulder", "right_hip"], ["right_knee", "right_hip"]),
        a(["right_shoulder", "right_hip"], ["right_ankle", "right_knee"]),
        a(["right_shoulder", "right_hip"], ["left_knee", "left_hip"]),
        a(["right_shoulder", "right_hip"], ["left_ankle", "left_knee"]),
        a(["left_hip", "left_knee"], ["right_hip", "right_knee"]),
        a(["left_hip", "left_knee"], ["right_ankle", "right_knee"]),
        a(["left_hip", "left_knee"], ["left_ankle", "left_knee"]),
        a(["right_hip", "right_knee"], ["right_ankle", "right_knee"]),
        a(["right_hip", "right_knee"], ["left_ankle", "left_knee"]),
        a(["left_hip", "right_elbow"], ["left_knee", "right_elbow"]),
        a(["right_hip", "right_elbow"], ["right_knee", "right_elbow"]),
        a(["right_hip", "left_elbow"], ["right_knee", "left_elbow"]),
        a(["left_hip", "left_elbow"], ["left_knee", "left_elbow"]),
    ]


def _gait_entries() -> list[tuple[str, str, str]]:
    """Returns (code, meaning, embed_coor slice used for each limb signal)."""
    # slice notes from gait_laging() — norms of embed_coor columns
    slices = {
        "r_ankle": "embed_coor[:,32:34]",
        "l_ankle": "embed_coor[:,30:32]",
        "r_knee": "embed_coor[:,28:30]",
        "l_knee": "embed_coor[:,26:28]",
        "r_hip": "embed_coor[:,26:28]",  # as in source (overlaps l_knee)
        "l_hip": "embed_coor[:,24:26]",
        "r_wrist": "embed_coor[:,22:24]",
        "l_wrist": "embed_coor[:,20:22]",
        "r_elbow": "embed_coor[:,18:20]",
        "l_elbow": "embed_coor[:,16:18]",
        "r_shoulder": "embed_coor[:,14:16]",
        "l_shoulder": "embed_coor[:,12:14]",
    }

    pairs = [
        ("r_wrist", "l_wrist"),
        ("r_elbow", "l_elbow"),
        ("r_ankle", "l_ankle"),
        ("r_shoulder", "l_shoulder"),
        ("r_knee", "l_knee"),
        ("r_hip", "l_hip"),
        ("r_wrist", "l_ankle"),
        ("r_wrist", "r_ankle"),
        ("l_wrist", "l_ankle"),
        ("l_wrist", "r_ankle"),
        ("r_wrist", "l_elbow"),
        ("r_wrist", "r_elbow"),
        ("l_wrist", "l_elbow"),
        ("l_wrist", "r_elbow"),
        ("r_wrist", "l_shoulder"),
        ("r_wrist", "r_shoulder"),
        ("l_wrist", "l_shoulder"),
        ("l_wrist", "r_shoulder"),
        ("r_wrist", "l_knee"),
        ("r_wrist", "r_knee"),
        ("l_wrist", "l_knee"),
        ("l_wrist", "r_knee"),
        ("r_wrist", "l_hip"),
        ("r_wrist", "r_hip"),
        ("l_wrist", "l_hip"),
        ("l_wrist", "r_hip"),
        ("r_elbow", "l_ankle"),
        ("r_elbow", "r_ankle"),
        ("l_elbow", "l_ankle"),
        ("l_elbow", "r_ankle"),
        ("r_elbow", "l_shoulder"),
        ("r_elbow", "r_shoulder"),
        ("l_elbow", "l_shoulder"),
        ("l_elbow", "r_shoulder"),
        ("r_elbow", "l_knee"),
        ("r_elbow", "r_knee"),
        ("l_elbow", "l_knee"),
        ("l_elbow", "r_knee"),
        ("r_elbow", "l_hip"),
        ("r_elbow", "r_hip"),
        ("l_elbow", "l_hip"),
        ("l_elbow", "r_hip"),
        ("r_ankle", "l_shoulder"),
        ("r_ankle", "r_shoulder"),
        ("l_ankle", "l_shoulder"),
        ("l_ankle", "r_shoulder"),
        ("r_ankle", "l_knee"),
        ("r_ankle", "r_knee"),
        ("l_ankle", "l_knee"),
        ("l_ankle", "r_knee"),
        ("r_ankle", "l_hip"),
        ("r_ankle", "r_hip"),
        ("l_ankle", "l_hip"),
        ("l_ankle", "r_hip"),
        ("r_shoulder", "l_knee"),
        ("r_shoulder", "r_knee"),
        ("l_shoulder", "l_knee"),
        ("l_shoulder", "r_knee"),
        ("r_shoulder", "l_hip"),
        ("r_shoulder", "r_hip"),
        ("l_shoulder", "l_hip"),
        ("l_shoulder", "r_hip"),
        ("r_knee", "l_hip"),
        ("r_knee", "r_hip"),
        ("l_knee", "l_hip"),
        ("l_knee", "r_hip"),
    ]
    out = []
    for a, b in pairs:
        code = f"gait_xcorr(norm({a}), norm({b}))"
        meaning = (
            f"Normalized cross-correlation (×100) between "
            f"{slices[a]} and {slices[b]} trajectory norms over the gait cycle"
        )
        out.append((code, meaning, f"{slices[a]} vs {slices[b]}"))
    return out


def build_rows() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    idx = 0

    for lm_i, lm in enumerate(LANDMARKS):
        for comp, suf in zip(("x", "y"), COORD_SUFFIX):
            rows.append(
                {
                    "index": idx,
                    "block": "embed_coors",
                    "code": f"norm_coord_{lm}{suf}",
                    "meaning": (
                        f"Normalized {comp.upper()} of landmark '{lm}' "
                        f"(YOLOv8 index {lm_i}); hip-centered, scale-normalized, ×1e5"
                    ),
                }
            )
            idx += 1

    for pair_i, (src, dst) in enumerate(_dist_entries()):
        for comp, sub in (("dx", "_x"), ("dy", "_y")):
            rows.append(
                {
                    "index": idx,
                    "block": "dis_embedders",
                    "code": f"dis_{pair_i:02d}{sub}",
                    "meaning": (
                        f"Signed {comp.upper()} of vector from '{src}' to '{dst}' "
                        f"(pair {pair_i} in distance embedding)"
                    ),
                }
            )
            idx += 1

    for ang_i, (v1, v2) in enumerate(_ang_entries()):
        v1s, v1e = v1
        v2s, v2e = v2
        rows.append(
            {
                "index": idx,
                "block": "ang_embedders",
                "code": f"ang_{ang_i:02d}",
                "meaning": (
                    f"Signed angle (degrees) between vector {v1s}→{v1e} and {v2s}→{v2e}"
                ),
            }
        )
        idx += 1

    for gait_i, (code, meaning, slices) in enumerate(_gait_entries()):
        rows.append(
            {
                "index": idx,
                "block": "gait_phases",
                "code": f"gait_{gait_i:02d}",
                "meaning": meaning,
                "slices": slices,
            }
        )
        idx += 1

    return rows


def main() -> None:
    rows = build_rows()
    assert len(rows) == 238, len(rows)
    out_md = r"c:\Users\Olive\Desktop\Nature_Style\code_video\utils\feature_index_table_238.md"
    out_csv = r"c:\Users\Olive\Desktop\Nature_Style\code_video\utils\feature_index_table_238.csv"

    import csv

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["index", "block", "code", "meaning"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Feature index table (238 features)",
        "",
        "Source: `utili_Getpose_v4.py` → `GetAllFeatures()` concatenation order:",
        "`embed_coors` (34) + `dis_embedders` (106) + `ang_embedders` (32) + `gait_phases` (66).",
        "",
        "| Index | Block | Code | Meaning |",
        "|------:|-------|------|---------|",
    ]
    for r in rows:
        meaning = str(r["meaning"]).replace("|", "\\|")
        lines.append(f"| {r['index']} | {r['block']} | `{r['code']}` | {meaning} |")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(rows)} rows to {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
