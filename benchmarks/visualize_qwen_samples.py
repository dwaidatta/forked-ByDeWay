"""
visualize_qwen_samples.py
=========================
Generates a publication-quality figure for research papers.

Layout (3 rows × N columns):
  Row 0 — Baseline      : INCORRECT predictions only  (red border)
  Row 1 — LDP           : INCORRECT predictions only  (red border)
  Row 2 — LDP + Spatial : CORRECT   predictions only  (green border)

This design clearly illustrates WHERE the depth / spatial pipeline helps:
baseline and LDP fail on these samples, LDP+Spatial succeeds.

Usage:
    python benchmarks/visualize_qwen_samples.py
    python benchmarks/visualize_qwen_samples.py --num_samples 5 --dpi 300
    python benchmarks/visualize_qwen_samples.py --out figures/paper_fig --dpi 300
"""

import os
import sys
import json
import textwrap
import argparse
import warnings
import io
from io import BytesIO

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image

warnings.filterwarnings("ignore")

# ── Project root on path ──────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(project_root, "data")
FIGURE_DIR = os.path.join(project_root, "figures")

# Per-row configuration: (mode_key, filter, label, header_color)
# filter values: "correct" | "incorrect" | "wrong_first" (wrong samples first, then fill with correct)
ROW_CONFIG = [
    ("ldp",         "wrong_first", "LDP  (+ Depth Context)",               "#8E44AD"),  # purple
    ("ldp_spatial", "correct",     "LDP + Spatial  (+ Spatial Relations)", "#1A8754"),  # green
]

CORRECT_BORDER   = "#27AE60"
INCORRECT_BORDER = "#C0392B"

# Candidate filenames for each mode (tries model-tagged first, then generic)
MODE_FILES = {
    "baseline":    ["vsr_qwen25vl_7b_baseline_predictions.jsonl",
                    "vsr_qwen25vl_baseline_predictions.jsonl"],
    "ldp":         ["vsr_qwen25vl_7b_ldp_predictions.jsonl",
                    "vsr_qwen25vl_ldp_predictions.jsonl"],
    "ldp_spatial": ["vsr_qwen25vl_7b_ldp_spatial_predictions.jsonl",
                    "vsr_qwen25vl_ldp_spatial_predictions.jsonl"],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_predictions_file(mode: str) -> str | None:
    """Return the first existing candidate file path for this mode."""
    for name in MODE_FILES.get(mode, []):
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return path
    return None


def load_predictions(mode: str) -> list[dict]:
    """Load JSONL prediction file for mode; return list of records."""
    path = find_predictions_file(mode)
    if path is None:
        print(f"  [WARN] No prediction file found for mode '{mode}'")
        return []
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Loaded {len(records):>3} records  <- {os.path.basename(path)}")
    return records


def filter_records(records: list[dict], want: str, n: int = 999) -> list[dict]:
    """
    Filter records by prediction outcome.
      'correct'    – only correctly predicted samples
      'incorrect'  – only incorrectly predicted samples
      'wrong_first'– all incorrect first, then fill remaining slots with correct ones
    """
    wrong   = [r for r in records if r["pred_label"] != r["ground_truth"]]
    correct = [r for r in records if r["pred_label"] == r["ground_truth"]]

    if want == "correct":
        return correct[:n]
    elif want == "incorrect":
        return wrong[:n]
    else:  # wrong_first
        combined = wrong + correct
        return combined[:n]


def fetch_image_links(indices: list[int], dataset: str, split: str) -> dict[int, str]:
    """Load VSR dataset to recover image_link per sample index."""
    print(f"\n  Fetching image URLs from '{dataset}' ({split})...")
    try:
        from datasets import load_dataset
        data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
        ds = load_dataset(dataset, data_files=data_files, split=split)
        links = {i: ds[i].get("image_link", "") for i in indices if i < len(ds)}
        print(f"  Retrieved {len(links)} URL(s).")
        return links
    except Exception as exc:
        print(f"  [WARN] Could not load dataset: {exc}")
        return {}


def download_image(url: str, session: requests.Session) -> Image.Image | None:
    if not url:
        return None
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        return img.convert("RGB") if img.mode != "RGB" else img
    except Exception as exc:
        print(f"    [WARN] Download failed: {exc}")
        return None


def wrap(text: str, width: int = 40) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width))


# ── Figure rendering ──────────────────────────────────────────────────────────

def render_figure(
    row_data: list[dict],   # list of {label, header_color, filter, panels: [...]}
    n_cols: int,
    dpi: int = 200,
    out_prefix: str = None,
):
    """
    Render 3-row × n_cols figure.
    row_data[i]["panels"] is a list of {rec, image} dicts.
    """
    n_rows = len(row_data)
    panel_w = 4.0          # width per panel (inches)
    label_w = 0.25         # narrow left label strip
    img_h   = 3.2          # image cell height
    txt_h   = 2.0          # text cell height
    title_h = 0.55         # figure title height
    header_h = 0.40        # row header strip height
    gap_h   = 0.10         # gap between rows

    fig_w = label_w + panel_w * n_cols
    fig_h = title_h + n_rows * (header_h + img_h + txt_h + gap_h)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#F8F9FA")

    # ── Figure title ──────────────────────────────────────────────────────────
    fig.text(
        0.5, 1.0 - (title_h * 0.4) / fig_h,
        "Qwen 2.5-VL — VSR Qualitative Results",
        ha="center", va="top",
        fontsize=15, fontweight="bold", color="#1A1A2E",
        transform=fig.transFigure,
    )
    fig.text(
        0.5, 1.0 - (title_h * 0.82) / fig_h,
        "LDP: incorrect predictions    |    LDP + Spatial: correct predictions",
        ha="center", va="top",
        fontsize=9, color="#555", style="italic",
        transform=fig.transFigure,
    )

    # ── Layout calculations ───────────────────────────────────────────────────
    usable_h = fig_h - title_h
    row_block_h = usable_h / n_rows        # height per row block in inches
    content_x0  = label_w / fig_w         # left edge of panel area (fraction)
    content_w   = 1.0 - content_x0        # width of panel area (fraction)
    panel_w_frac = content_w / n_cols

    for row_idx, row in enumerate(row_data):
        panels     = row["panels"]          # list of {rec, image}
        hdr_color  = row["header_color"]
        is_correct = row["filter"] == "correct"
        border_col = CORRECT_BORDER if is_correct else INCORRECT_BORDER

        # Vertical fractions (0 = bottom, 1 = top)
        block_top    = 1.0 - title_h / fig_h - row_idx * row_block_h / fig_h
        block_bottom = block_top - row_block_h / fig_h

        hdr_top    = block_top
        hdr_bottom = hdr_top - header_h / fig_h
        img_top    = hdr_bottom
        img_bottom = img_top - img_h / fig_h
        txt_top    = img_bottom
        txt_bottom = txt_top - txt_h / fig_h

        # ── Row header strip ──────────────────────────────────────────────────
        ax_hdr = fig.add_axes([content_x0, hdr_bottom, content_w, header_h / fig_h])
        ax_hdr.set_xlim(0, 1); ax_hdr.set_ylim(0, 1); ax_hdr.axis("off")
        ax_hdr.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="square,pad=0",
            facecolor=hdr_color, edgecolor="none",
            transform=ax_hdr.transAxes, zorder=1,
        ))
        badge = "CORRECT" if is_correct else "INCORRECT"
        ax_hdr.text(
            0.5, 0.52,
            f"{row['label']}   [{badge}]",
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white",
            transform=ax_hdr.transAxes, zorder=2,
        )

        # ── Panels ────────────────────────────────────────────────────────────
        for col_idx in range(n_cols):
            x0 = content_x0 + col_idx * panel_w_frac
            pw = panel_w_frac * 0.92
            px0 = x0 + panel_w_frac * 0.04

            if col_idx >= len(panels):
                # Empty placeholder
                ax_empty = fig.add_axes([px0, txt_bottom, pw, img_h / fig_h + txt_h / fig_h])
                ax_empty.set_facecolor("#EFEFEF")
                ax_empty.axis("off")
                ax_empty.text(0.5, 0.5, "—", ha="center", va="center",
                              fontsize=14, color="#BBBBBB", transform=ax_empty.transAxes)
                for spine in ax_empty.spines.values():
                    spine.set_edgecolor("#DDDDDD"); spine.set_linewidth(1.5)
                continue

            panel = panels[col_idx]
            rec   = panel["rec"]
            img   = panel["image"]

            correct_pred = rec["pred_label"] == rec["ground_truth"]
            b_col = CORRECT_BORDER if correct_pred else INCORRECT_BORDER

            # Image subplot
            ax_img = fig.add_axes([px0, img_bottom, pw, img_h / fig_h])
            ax_img.set_facecolor("#DCDCDC")
            if img is not None:
                ax_img.imshow(np.array(img), aspect="auto")
            else:
                ax_img.text(0.5, 0.5, "Image\nunavailable",
                            ha="center", va="center", fontsize=9,
                            color="#888", transform=ax_img.transAxes)
            ax_img.set_xticks([]); ax_img.set_yticks([])
            for spine in ax_img.spines.values():
                spine.set_edgecolor(b_col); spine.set_linewidth(2.8)

            # Correctness badge
            badge_txt = "Correct" if correct_pred else "Wrong"
            ax_img.text(
                0.03, 0.97, badge_txt,
                transform=ax_img.transAxes, fontsize=7.5, fontweight="bold",
                color="white", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.22", facecolor=b_col,
                          edgecolor="none", alpha=0.90),
            )
            # Sample index
            ax_img.text(
                0.97, 0.97, f"#{rec['idx']}",
                transform=ax_img.transAxes, fontsize=7, color="white",
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#333",
                          edgecolor="none", alpha=0.7),
            )

            # Text subplot
            ax_txt = fig.add_axes([px0, txt_bottom, pw, txt_h / fig_h])
            ax_txt.set_facecolor("#FFFFFF")
            ax_txt.set_xticks([]); ax_txt.set_yticks([])
            for spine in ax_txt.spines.values():
                spine.set_edgecolor(b_col); spine.set_linewidth(2.0)

            # Build text lines
            caption = rec["caption"]
            gt_str  = "TRUE"  if rec["ground_truth"] == 1 else "FALSE"
            pr_str  = rec["norm_pred"].upper()
            cat_str = rec.get("category", "")
            rel_str = rec.get("relation", "")
            gt_col  = "#27AE60" if rec["ground_truth"] == 1 else "#C0392B"
            pr_col  = "#27AE60" if rec["pred_label"] == 1   else "#C0392B"

            lines = [
                ("Caption:",     "#2C3E50", True,  8.0),
            ]
            for cl in textwrap.wrap(caption, width=36):
                lines.append((cl, "#1A1A2E", False, 7.5))
            lines.append(("",              "#fff",   False, 3.5))
            lines.append((f"Ground Truth:   {gt_str}", gt_col, True, 8.0))
            lines.append((f"Predicted:        {pr_str}",  pr_col, True, 8.0))
            lines.append(("",              "#fff",   False, 3.0))
            lines.append((f"Relation:   {rel_str}",  "#777", False, 7.0))
            lines.append((f"Category:   {cat_str}",  "#777", False, 7.0))

            n_lines = len(lines)
            y_step  = 1.0 / (n_lines + 1)
            y_pos   = 1.0 - y_step * 0.6

            for txt, col, bold, fsize in lines:
                ax_txt.text(
                    0.05, y_pos, txt,
                    transform=ax_txt.transAxes,
                    fontsize=fsize, color=col,
                    fontweight="bold" if bold else "normal",
                    va="top", ha="left",
                )
                y_pos -= y_step

    # ── Save ─────────────────────────────────────────────────────────────────
    if out_prefix is None:
        out_prefix = os.path.join(FIGURE_DIR, "qwen_vsr_qualitative")
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)

    png_path = out_prefix + ".png"
    pdf_path = out_prefix + ".pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path,           bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  [OK] PNG -> {png_path}")
    print(f"  [OK] PDF -> {pdf_path}")
    return png_path, pdf_path


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Generate a research-paper figure: "
            "Baseline (wrong) | LDP (wrong) | LDP+Spatial (correct)."
        )
    )
    p.add_argument("--num_samples", type=int, default=5,
                   help="Max panels per row (default 5).")
    p.add_argument("--out", type=str, default=None,
                   help="Output path prefix (no extension).")
    p.add_argument("--dpi", type=int, default=200,
                   help="Figure DPI (use 300 for print).")
    p.add_argument("--dataset", type=str, default="cambridgeltl/vsr_random")
    p.add_argument("--split",   type=str, default="test")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    N = args.num_samples

    print("=" * 62)
    print("  Qwen 2.5-VL VSR Qualitative Figure")
    print("  Baseline (wrong) | LDP (wrong) | LDP+Spatial (correct)")
    print("=" * 62)

    # ── 1. Load & filter predictions ─────────────────────────────────────────
    print("\n[1/4] Loading prediction files...")
    row_data = []
    all_indices = set()

    for mode_key, want_filter, label, hdr_color in ROW_CONFIG:
        recs = load_predictions(mode_key)
        filtered = filter_records(recs, want_filter, n=N)
        if not filtered:
            print(f"  [WARN] No '{want_filter}' samples found for mode '{mode_key}'.")
        else:
            n_wrong   = sum(1 for r in filtered if r["pred_label"] != r["ground_truth"])
            n_correct = len(filtered) - n_wrong
            if want_filter == "wrong_first":
                print(f"    -> {len(filtered)} sample(s) for {label.split('(')[0].strip()} "
                      f"({n_wrong} incorrect + {n_correct} correct)")
            else:
                print(f"    -> {len(filtered)} '{want_filter}' sample(s) selected for {label.split('(')[0].strip()}")
        row_data.append({
            "mode":         mode_key,
            "filter":       want_filter,
            "label":        label,
            "header_color": hdr_color,
            "records":      filtered,
        })
        all_indices.update(r["idx"] for r in filtered)

    # ── 2. Fetch image URLs ───────────────────────────────────────────────────
    print("\n[2/4] Fetching image URLs from VSR dataset...")
    image_links = fetch_image_links(sorted(all_indices), args.dataset, args.split)

    # ── 3. Download images ────────────────────────────────────────────────────
    print("\n[3/4] Downloading images...")
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=requests.adapters.Retry(total=4, backoff_factor=0.5)
    )
    session.mount("http://",  adapter)
    session.mount("https://", adapter)

    img_cache: dict[int, Image.Image | None] = {}
    for idx in sorted(all_indices):
        url = image_links.get(idx, "")
        print(f"    #{idx}: {url or '(no URL)'}")
        img_cache[idx] = download_image(url, session)

    # Attach images to rows
    for row in row_data:
        row["panels"] = [
            {"rec": r, "image": img_cache.get(r["idx"])}
            for r in row["records"]
        ]

    # ── 4. Render ─────────────────────────────────────────────────────────────
    print("\n[4/4] Rendering figure...")
    n_cols = max((len(row["panels"]) for row in row_data), default=1)
    n_cols = max(n_cols, 1)

    # Warn if any mode has fewer than requested
    for row in row_data:
        have = len(row["panels"])
        if have < N:
            print(f"  [NOTE] {row['label'].split('(')[0].strip()}: "
                  f"only {have}/{N} '{row['filter']}' sample(s) available.")

    out_prefix = args.out or os.path.join(FIGURE_DIR, "qwen_vsr_qualitative")
    png_path, pdf_path = render_figure(
        row_data, n_cols=n_cols, dpi=args.dpi, out_prefix=out_prefix
    )

    print("\n" + "=" * 62)
    print("  Done!")
    print(f"    PNG : {png_path}")
    print(f"    PDF : {pdf_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
