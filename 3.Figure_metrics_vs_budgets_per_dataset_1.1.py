#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create 1x2 combined figures (Accuracy | Information Loss) vs Budget for each dataset.
- One PDF per dataset, saved in OUTPUT_DIR
- Distinct markers + linestyles per method; colors come from Matplotlib's default cycle
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import glob

# -------- Font & style knobs (adjust here) --------
base_font_size = 18
title_size     = base_font_size + 2
figure_title_size = base_font_size + 4
label_size     = base_font_size
tick_size      = base_font_size - 2
legend_size    = base_font_size - 1
legend_title_size = base_font_size
marker_size    = 8
line_width     = 2.5
grid_linewidth = 0.8

plt.rcParams.update({
    "font.size": base_font_size,
    "axes.titlesize": title_size,
    "axes.labelsize": label_size,
    "legend.fontsize": legend_size
})

# ========= EDIT THESE =========
CSV_PATH     = r""  # leave empty "" to auto-pick latest
OUTPUT_DIR   = r"./fig_out"
FULL_METRICS = False
# ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

if CSV_PATH and os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    exp_files = sorted(glob.glob("tables/wilcoxon_per_budget-exp_*.csv"),
                       key=lambda f: os.path.getmtime(f))
    if not exp_files:
        raise FileNotFoundError("No CSV found. Set CSV_PATH or ensure results/ has matching files.")
    latest_file = exp_files[-1]
    df = pd.read_csv(latest_file)


def resolve_column(possible_names, df_cols, required=True):
    lowered = {c.lower(): c for c in df_cols}
    for name in possible_names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    if required:
        raise ValueError(
            f"Required column not found. Tried: {possible_names}. "
            f"Available: {list(df_cols)}"
        )
    return None

dataset_col = resolve_column(["dataset", "data", "ds", "dataset_name"], df.columns)
budget_col  = resolve_column(["budget", "budget_value", "b", "epsilon", "eps"], df.columns)
metric_col  = resolve_column(["metric", "measure", "metric_name"], df.columns)

# find method columns
mean_cols = [c for c in df.columns if re.match(r"(?i)^mean[\s_\-].+", c)]
if not mean_cols:
    raise ValueError("No method columns found. Expect columns like 'mean_ga', 'mean_das', ...")

use_cols = [dataset_col, budget_col, metric_col] + mean_cols
df = df[use_cols].copy()

df[budget_col] = pd.to_numeric(df[budget_col], errors="coerce")
df = df.dropna(subset=[dataset_col, budget_col, metric_col])

def norm_metric(x: str) -> str:
    return str(x).strip().lower().replace(" ", "_")

df[metric_col] = df[metric_col].apply(norm_metric)

ACC_NAMES = {"accuracy", "acc"}
IL_NAMES  = {"information_loss", "info_loss", "il", "loss"}

markers = ["o", "s", "^", "D", "v", ">", "<", "P", "X", "*", "h"]
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 1))]

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", str(name))
    return name.strip("_").lower()

def extract_method_name(col: str) -> str:
    return re.sub(r"(?i)^mean[\s_\-]*", "", col).strip().upper()

def build_style(methods):
    style = {}
    for i, m in enumerate(sorted(methods)):
        style[m] = {
            "marker": markers[i % len(markers)],
            "linestyle": linestyles[i % len(linestyles)],
        }
    return style

def plot_metric(ax, subset, budget_col, ycols, styles, title, ylabel=None, add_legend=True):
    for col in ycols:
        method = extract_method_name(col)
        ax.plot(
            subset[budget_col],
            pd.to_numeric(subset[col], errors="coerce"),
            marker=styles[method]["marker"],
            linestyle=styles[method]["linestyle"],
            label=method,
            linewidth=line_width,
            markersize=marker_size,
        )
    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel("Budget", fontsize=label_size)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_size)
    ax.grid(True, linestyle=":", linewidth=grid_linewidth)
    ax.tick_params(axis="both", labelsize=tick_size)
    if add_legend:
        leg = ax.legend(title="Method", loc="best", fontsize=legend_size)
        if leg and leg.get_title():
            leg.get_title().set_fontsize(legend_title_size)


# ---------- Per-dataset drawing ----------
for ds, g in df.groupby(dataset_col):
    methods = [extract_method_name(c) for c in mean_cols]
    styles = build_style(methods)

    if not FULL_METRICS:
        # 1x2 layout: Accuracy | Info Loss
        g_acc = g[g[metric_col].isin(ACC_NAMES)].sort_values(budget_col)
        g_il  = g[g[metric_col].isin(IL_NAMES)].sort_values(budget_col)

        if g_acc.empty and g_il.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, constrained_layout=True)

        if not g_acc.empty:
            plot_metric(axes[0], g_acc, budget_col, mean_cols, styles,
                        f"Accuracy vs Budget", "Accuracy", add_legend=True)
        else:
            axes[0].set_visible(False)

        if not g_il.empty:
            plot_metric(axes[1], g_il, budget_col, mean_cols, styles,
                        f"Information Loss vs Budget", "Information Loss",add_legend=True)
        else:
            axes[1].set_visible(False)

        out_path = os.path.join(OUTPUT_DIR, f"{sanitize_filename(ds)}_acc_il_vs_budget.pdf")
        # fig.suptitle(f"Dataset: {ds}", fontsize=figure_title_size)
        fig.savefig(out_path, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"Saved: {out_path}")

    else:
        # 2x2 layout of metrics, with Info Loss forced last
        metrics_all = list(dict.fromkeys(g[metric_col].unique()))
        il_present = [m for m in metrics_all if m in IL_NAMES]
        others = [m for m in metrics_all if m not in IL_NAMES]

        ordered = others[:3]  # take at most 3 others
        if il_present:
            ordered.append(il_present[0])  # put IL as last slot
        else:
            ordered = ordered[:4]  # just 4 metrics max

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300, constrained_layout=True)
        axes_flat = axes.ravel()

        for i, m in enumerate(ordered):
            ax = axes_flat[i]
            sub = g[g[metric_col] == m].sort_values(budget_col)
            if sub.empty:
                ax.set_visible(False)
                continue
            title_text = m.replace("_", " ").title()
            ylabel = "Accuracy" if m in ACC_NAMES else ("Information Loss" if m in IL_NAMES else title_text)
            plot_metric(ax, sub, budget_col, mean_cols, styles,
                        f"{title_text} vs Budget", ylabel,add_legend=True)

        # hide unused panels
        for j in range(len(ordered), 4):
            axes_flat[j].set_visible(False)

        out_path = os.path.join(OUTPUT_DIR, f"{sanitize_filename(ds)}_metrics_2x2.pdf")
        # fig.suptitle(f"Dataset: {ds}", fontsize=figure_title_size)
        fig.savefig(out_path, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"Saved: {out_path}")
