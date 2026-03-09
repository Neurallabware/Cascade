#!/usr/bin/env python3
"""
Plot Figure 3 of the CASCADE paper: "Generalization across datasets"

Reproduces all 6 panels:
  a) Correlation heatmap (training × test datasets)
  b) Box plots of correlation distribution per model
  c) Error heatmap
  d) Box/scatter plots of error distribution
  e) Bias heatmap
  f) Box/scatter plots of bias distribution

Loads pre-computed results from fig3_benchmark_results.npz
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

# Project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ── Load results ──────────────────────────────────────────────────────────────
def load_data():
    results_path = os.path.join(project_root, 'notebooks', 'fig3_benchmark_results.npz')
    if not os.path.exists(results_path):
        results_path = os.path.join(project_root, 'notebooks', 'fig3_benchmark_results_partial.npz')
        if not os.path.exists(results_path):
            print("ERROR: No benchmark results found. Run fig3_compute_benchmark.py first.")
            sys.exit(1)
    return np.load(results_path, allow_pickle=True)


# ── Constants ─────────────────────────────────────────────────────────────────
N_DATASETS = 26   # 21 exc + 5 inh
N_EXC = 21
N_INH = 5

# Row indices in the data matrix: 0-25 = single datasets, 26=NAOMi, 27=GlobalEXC, 28=GlobalINH
IDX_NAOMI = N_DATASETS
IDX_EXC = N_DATASETS + 1
IDX_INH = N_DATASETS + 2

# Display order: rows 1-21 (exc), SIM, EXC, rows 22-26 (inh), INH
ROW_ORDER = list(range(N_EXC)) + [IDX_NAOMI, IDX_EXC] + list(range(N_EXC, N_DATASETS)) + [IDX_INH]
N_ROWS = len(ROW_ORDER)

# Row number labels (right side of heatmap, matching paper)
ROW_NUMBERS = list(range(1, N_EXC + 1)) + ['SIM', 'EXC'] + list(range(22, 27)) + ['INH']

# Row name labels (left side of panel a, colored by indicator type)
ROW_NAMES = [
    'OGB-1, mouse',       # 1
    'OGB-1, mouse',       # 2
    'Cal-520, mouse',     # 3
    'OGB-1, fish',        # 4
    'Cal-520, fish',      # 5
    'GCaMP6f, fish',      # 6
    'GCaMP6f, fish',      # 7
    'GCaMP6f, fish',      # 8
    'GCaMP6f, mouse',     # 9
    'GCaMP6f, mouse',     # 10
    'GCaMP6f, mouse',     # 11
    'GCaMP6s, mouse',     # 12
    'GCaMP6s, mouse',     # 13
    'GCaMP6s, mouse',     # 14
    'GCaMP6s, mouse',     # 15
    'GCaMP6s, mouse',     # 16
    'GCaMP5k, mouse',     # 17
    'R-CaMP1.07, mouse',  # 18
    'R-CaMP1.07, mouse',  # 19
    'jRCaMP1a, mouse',    # 20
    'jRGECO1a, mouse',    # 21
    'NAOMi sim. data',     # SIM
    'Global EXC model',    # EXC
    'OGB-1, mouse SOM',   # 22
    'OGB-1, mouse PV',    # 23
    'GCaMP6f, mouse PV',  # 24
    'GCaMP6f, mouse SOM', # 25
    'GCaMP6f, mouse VIP', # 26
    'Global INH model',    # INH
]

# Column labels (top of heatmap)
COL_NUMBERS = list(range(1, N_EXC + 1)) + ['SIM'] + list(range(22, 27))

# ── Colors matching the paper ─────────────────────────────────────────────────
# Row colors by indicator type (matching the paper's color coding)
# Paper: OGB/Cal520 = black/dark, fish = blue, GCaMP6f = blue, GCaMP6s = dark blue,
# other = red, interneurons = red/dark red

def _row_color(display_idx):
    """Color for each row based on display position."""
    if display_idx < N_EXC:
        ds = display_idx + 1
        if ds <= 3:                          # OGB-1, Cal-520 (synth. dyes)
            return '#444444'
        elif ds in [4, 5]:                   # zebrafish
            return '#2196F3'
        elif ds in [6, 7, 8]:               # GCaMP6f, fish
            return '#1976D2'
        elif ds in [9, 10, 11]:             # GCaMP6f, mouse
            return '#388E3C'
        elif ds in [12, 13, 14, 15, 16]:    # GCaMP6s, mouse
            return '#F57C00'
        elif ds == 17:                       # GCaMP5k
            return '#D32F2F'
        elif ds in [18, 19]:                # R-CaMP
            return '#C62828'
        elif ds == 20:                       # jRCaMP1a
            return '#AD1457'
        elif ds == 21:                       # jRGECO1a
            return '#6A1B9A'
        return '#444444'
    elif display_idx == N_EXC:     # SIM (NAOMi)
        return '#607D8B'
    elif display_idx == N_EXC + 1: # EXC
        return '#1565C0'
    elif display_idx < N_EXC + 2 + N_INH:  # interneurons 22-26
        return '#B71C1C'
    else:  # INH
        return '#880E4F'


# Column category spans for bottom labels (matching paper)
COL_GROUPS = [
    (0, 3, 'Synth.', '#444444'),
    (3, 5, 'Zebrafish', '#2196F3'),
    (5, 11, 'GCaMP6f', '#388E3C'),
    (11, 16, 'GCaMP6s', '#F57C00'),
    (16, 21, 'Other indicators', '#C62828'),
    (21, 26, 'Interneurons', '#B71C1C'),
]


def reorder(matrix):
    """Reorder matrix rows from data order to display order."""
    return matrix[ROW_ORDER, :]


# ── Panel plotting functions ──────────────────────────────────────────────────

def plot_heatmap_squares(ax, mat, cmap, norm, title, show_names=True, show_col_groups=True):
    """
    Generic heatmap with sized colored squares.

    In the paper, square size and color both encode the value.
    """
    ax.set_facecolor('#F5F5F5')

    for i in range(N_ROWS):
        for j in range(N_DATASETS):
            val = mat[i, j]
            if np.isnan(val):
                continue

            # Map value to [0,1] range for size
            norm_val = np.clip(norm(val), 0, 1)
            size = max(0.15, norm_val) * 0.88
            color = cmap(norm_val)

            rect = Rectangle(
                (j + 0.5 - size / 2, i + 0.5 - size / 2),
                size, size,
                facecolor=color,
                edgecolor='none',
            )
            ax.add_patch(rect)

    ax.set_xlim(0, N_DATASETS)
    ax.set_ylim(N_ROWS, 0)

    # Column labels at top — only the data column numbers
    col_ticks = np.arange(N_DATASETS) + 0.5
    # Map columns to their display numbers
    # Columns 0-20 = datasets 1-21, columns 21-25 = datasets 22-26
    col_labels = list(range(1, N_EXC + 1)) + list(range(22, 27))
    ax.set_xticks(col_ticks)
    ax.set_xticklabels(col_labels, fontsize=4.5, rotation=45, ha='left')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Right side: row number labels
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position('right')
    ax.set_yticks(np.arange(N_ROWS) + 0.5)
    ax.set_yticklabels([str(x) for x in ROW_NUMBERS], fontsize=5, ha='left')

    # Left side: dataset name labels (colored)
    if show_names:
        for i, name in enumerate(ROW_NAMES):
            color = _row_color(i)
            ax.text(-0.3, i + 0.5, name, fontsize=4.5, ha='right', va='center',
                    color=color, transform=ax.get_yaxis_transform())

    # Separator lines
    # Between excitatory and SIM/EXC
    ax.axhline(y=N_EXC, color='#AAAAAA', linewidth=0.5)
    # Between EXC and inhibitory
    ax.axhline(y=N_EXC + 2, color='#AAAAAA', linewidth=0.5)
    # Between exc and inh columns
    ax.axvline(x=N_EXC, color='#AAAAAA', linewidth=0.5)

    ax.set_title(title, fontsize=8, pad=20)

    # Column group labels at bottom
    if show_col_groups:
        for start, end, label, color in COL_GROUPS:
            mid = (start + end) / 2
            ax.text(mid, N_ROWS + 0.7, label, fontsize=4, ha='center', va='top',
                    color=color, fontweight='bold')

    # Grid lines (very faint)
    for i in range(N_ROWS + 1):
        ax.axhline(y=i, color='#E0E0E0', linewidth=0.2, zorder=0)
    for j in range(N_DATASETS + 1):
        ax.axvline(x=j, color='#E0E0E0', linewidth=0.2, zorder=0)


def plot_boxplots(ax, mat, title, xlabel, xlim, xscale='linear',
                  dashed_line_value=None, zero_line=False):
    """
    Horizontal box plots (one per row), matching paper style.
    """
    for i in range(N_ROWS):
        row_data = mat[i, :]
        valid = row_data[~np.isnan(row_data)]
        if len(valid) == 0:
            continue

        color = _row_color(i)

        bp = ax.boxplot(
            valid, positions=[i], widths=0.6,
            vert=False, patch_artist=True,
            showfliers=True,
            flierprops=dict(marker='.', markersize=1.5, markerfacecolor='gray',
                           markeredgecolor='gray'),
            medianprops=dict(color='black', linewidth=0.8),
            boxprops=dict(linewidth=0.4, edgecolor='gray'),
            whiskerprops=dict(linewidth=0.4, color='gray'),
            capprops=dict(linewidth=0.4, color='gray'),
        )
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.35)

    if dashed_line_value is not None:
        ax.axvline(x=dashed_line_value, color='black', linewidth=0.8,
                   linestyle='--', alpha=0.6, zorder=5)

    if zero_line:
        ax.axvline(x=0, color='#888888', linewidth=0.4, linestyle='-')

    if xscale == 'log':
        ax.set_xscale('log')

    ax.set_xlim(xlim)
    ax.set_ylim(N_ROWS - 0.5, -0.5)
    ax.set_xlabel(xlabel, fontsize=6)
    ax.tick_params(axis='x', labelsize=5)
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)

    # Separator lines
    ax.axhline(y=N_EXC, color='#AAAAAA', linewidth=0.5)
    ax.axhline(y=N_EXC + 2, color='#AAAAAA', linewidth=0.5)

    # Add scale reference labels for y-axis
    for i, num in enumerate(ROW_NUMBERS):
        if isinstance(num, int) and num % 5 == 0:
            ax.text(-0.02, i, str(num), fontsize=3.5, ha='right', va='center',
                    transform=ax.get_yaxis_transform(), color='gray')


# ── Panel a: Correlation heatmap ──────────────────────────────────────────────
def plot_panel_a(ax, corr_matrix):
    mat = reorder(corr_matrix)
    norm = Normalize(vmin=0, vmax=1)
    plot_heatmap_squares(ax, mat, plt.cm.YlOrRd, norm,
                         'Correlation with ground truth')
    return ax


# ── Panel b: Correlation box plots ────────────────────────────────────────────
def plot_panel_b(ax, corr_matrix):
    mat = reorder(corr_matrix)
    # Get Global EXC median for dashed line
    exc_idx = ROW_ORDER.index(IDX_EXC)
    exc_median = np.nanmedian(mat[exc_idx, :])
    plot_boxplots(ax, mat, 'Correlation', 'Correlation',
                  xlim=(0, 1.05), dashed_line_value=exc_median)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    return ax


# ── Panel c: Error heatmap ────────────────────────────────────────────────────
def plot_panel_c(ax, error_matrix):
    mat = reorder(error_matrix)
    valid = mat[~np.isnan(mat)]
    vmax = np.percentile(valid, 95) if len(valid) > 0 else 5
    norm = Normalize(vmin=0, vmax=vmax)
    plot_heatmap_squares(ax, mat, plt.cm.YlOrRd, norm,
                         'Error of predictions', show_names=False)
    return ax


# ── Panel d: Error box plots ─────────────────────────────────────────────────
def plot_panel_d(ax, error_matrix):
    mat = reorder(error_matrix)
    exc_idx = ROW_ORDER.index(IDX_EXC)
    exc_median = np.nanmedian(mat[exc_idx, :])
    plot_boxplots(ax, mat, 'Error', 'Error',
                  xlim=(0.5, 60), xscale='log', dashed_line_value=exc_median)
    return ax


# ── Panel e: Bias heatmap ────────────────────────────────────────────────────
def plot_panel_e(ax, bias_matrix):
    mat = reorder(bias_matrix)
    valid = mat[~np.isnan(mat)]
    vabs = np.percentile(np.abs(valid), 95) if len(valid) > 0 else 3
    norm = Normalize(vmin=-vabs, vmax=vabs)
    plot_heatmap_squares(ax, mat, plt.cm.PuOr_r, norm,
                         'Bias of relative error', show_names=False)
    return ax


# ── Panel f: Bias box plots ──────────────────────────────────────────────────
def plot_panel_f(ax, bias_matrix):
    mat = reorder(bias_matrix)
    plot_boxplots(ax, mat, 'Bias', 'Bias',
                  xlim=(-3.5, 5.5), zero_line=True)
    ax.set_xticks([-1, 0, 1, 2, 3, 4, 5])
    return ax


# ── Individual panel saving ───────────────────────────────────────────────────
def save_individual_panels(output_dir, data):
    os.makedirs(output_dir, exist_ok=True)

    corr = data['corr_matrix']
    err = data['error_matrix']
    bias = data['bias_matrix']

    panels = [
        ('a', plot_panel_a, [corr], (9, 8)),
        ('b', plot_panel_b, [corr], (3.5, 8)),
        ('c', plot_panel_c, [err], (9, 8)),
        ('d', plot_panel_d, [err], (3.5, 8)),
        ('e', plot_panel_e, [bias], (9, 8)),
        ('f', plot_panel_f, [bias], (3.5, 8)),
    ]

    for name, fn, args, figsize in panels:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fn(ax, *args)
        fig.tight_layout()
        for ext in ['pdf', 'png']:
            path = os.path.join(output_dir, f'fig3_panel_{name}.{ext}')
            fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved panel {name}")


# ── Assembled Figure 3 ───────────────────────────────────────────────────────
def assemble_figure3(output_dir, data):
    os.makedirs(output_dir, exist_ok=True)

    corr = data['corr_matrix']
    err = data['error_matrix']
    bias = data['bias_matrix']

    # Paper layout: 3 rows × 2 columns
    # Left (wider): heatmaps (a, c, e)
    # Right (narrower): box plots (b, d, f)
    fig = plt.figure(figsize=(14, 18))

    gs = gridspec.GridSpec(
        3, 2,
        width_ratios=[3.5, 1],
        height_ratios=[1, 1, 1],
        wspace=0.08,
        hspace=0.25,
    )

    # Row 1: Correlation
    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel_a(ax_a, corr)
    ax_a.text(-0.12, 1.02, 'a', transform=ax_a.transAxes,
              fontsize=16, fontweight='bold', va='bottom')

    ax_b = fig.add_subplot(gs[0, 1])
    plot_panel_b(ax_b, corr)
    ax_b.text(-0.15, 1.02, 'b', transform=ax_b.transAxes,
              fontsize=16, fontweight='bold', va='bottom')

    # Row 2: Error
    ax_c = fig.add_subplot(gs[1, 0])
    plot_panel_c(ax_c, err)
    ax_c.text(-0.12, 1.02, 'c', transform=ax_c.transAxes,
              fontsize=16, fontweight='bold', va='bottom')

    ax_d = fig.add_subplot(gs[1, 1])
    plot_panel_d(ax_d, err)
    ax_d.text(-0.15, 1.02, 'd', transform=ax_d.transAxes,
              fontsize=16, fontweight='bold', va='bottom')

    # Row 3: Bias
    ax_e = fig.add_subplot(gs[2, 0])
    plot_panel_e(ax_e, bias)
    ax_e.text(-0.12, 1.02, 'e', transform=ax_e.transAxes,
              fontsize=16, fontweight='bold', va='bottom')

    ax_f = fig.add_subplot(gs[2, 1])
    plot_panel_f(ax_f, bias)
    ax_f.text(-0.15, 1.02, 'f', transform=ax_f.transAxes,
              fontsize=16, fontweight='bold', va='bottom')

    fig.suptitle(
        'Fig. 3 | Generalization across datasets',
        fontsize=13, fontweight='bold', y=0.995
    )

    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'Figure_3.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nAssembled Figure 3 saved to {output_dir}/Figure_3.{{pdf,png}}")


if __name__ == '__main__':
    data = load_data()
    output_dir = os.path.join(project_root, 'notebooks', 'figures')
    save_individual_panels(output_dir, data)
    assemble_figure3(output_dir, data)
