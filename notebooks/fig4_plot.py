#!/usr/bin/env python3
"""
Plot Figure 4 of the CASCADE paper: "Comparison with model-based algorithms"

Panels a-j reproduced using available algorithms (CASCADE + OASIS).
Panels requiring MLSpike/Peeling/CaImAn/Suite2p/Jewell&Witten are shown
as placeholders until those algorithms are installed and run.

Loads results from fig4_comparison_results.npz
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ── Algorithm style ───────────────────────────────────────────────────────────
ALGO_STYLE = {
    'cascade_tuned': {'color': '#1976D2', 'label': 'CASCADE tuned',  'ls': '-',  'lw': 2},
    'cascade_exc':   {'color': '#42A5F5', 'label': 'CASCADE EXC',    'ls': '--', 'lw': 1.5},
    'oasis':         {'color': '#FF9800', 'label': 'OASIS',          'ls': '-',  'lw': 1.5},
    'mlspike':       {'color': '#4CAF50', 'label': 'MLSpike tuned',  'ls': '-',  'lw': 1.5},
    'peeling':       {'color': '#F44336', 'label': 'Peeling tuned',  'ls': '-',  'lw': 1.5},
    'caiman':        {'color': '#9C27B0', 'label': 'CaImAn tuned',   'ls': '-',  'lw': 1.5},
    'suite2p':       {'color': '#795548', 'label': 'Suite2p tuned',  'ls': '-',  'lw': 1.5},
    'jewell':        {'color': '#E91E63', 'label': "Jewell&Witten",  'ls': '-',  'lw': 1.5},
}


def load_results():
    results_path = os.path.join(project_root, 'notebooks', 'fig4_comparison_results.npz')
    if not os.path.exists(results_path):
        print(f"ERROR: No results at {results_path}. Run fig4_compute_comparison.py first.")
        sys.exit(1)
    return np.load(results_path, allow_pickle=True)


# ── Panel b: Correlation heatmap ──────────────────────────────────────────────
def plot_panel_b(ax, data):
    """Heatmap rows=algorithms, columns=datasets, color=correlation at noise level 2."""
    algorithms = list(data['algorithms'])
    noise_levels = list(data['noise_levels'])
    nl2_idx = noise_levels.index(2) if 2 in noise_levels else 0
    display_idx = list(data['display_indices'])
    n_ds = len(display_idx)

    # Collect available algorithm data
    algo_names = []
    algo_data = []
    for algo in algorithms:
        key = f'{algo}_corr'
        if key in data:
            mat = data[key]
            if mat.ndim == 2 and mat.shape[0] > nl2_idx:
                algo_names.append(algo)
                algo_data.append(mat[nl2_idx, :])

    if not algo_data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    heatmap = np.array(algo_data)  # shape: (n_algos, n_datasets)
    n_algos = heatmap.shape[0]

    im = ax.imshow(heatmap, aspect='auto', cmap='YlOrRd',
                   vmin=0.3, vmax=0.9, interpolation='nearest')

    ax.set_yticks(range(n_algos))
    labels = [ALGO_STYLE.get(a, {}).get('label', a) for a in algo_names]
    colors = [ALGO_STYLE.get(a, {}).get('color', 'black') for a in algo_names]
    ax.set_yticklabels(labels, fontsize=6)
    for tick, color in zip(ax.get_yticklabels(), colors):
        tick.set_color(color)

    ax.set_xticks(range(n_ds))
    ax.set_xticklabels(display_idx, fontsize=4, rotation=90)
    ax.set_xlabel('Dataset index', fontsize=6)
    ax.set_title('Correlation with ground truth; noise level 2', fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02, aspect=15)
    cbar.ax.tick_params(labelsize=5)

    # Separate exc/inh columns
    # Find where inhibitory datasets start (index 22 in display)
    try:
        inh_start = [i for i, d in enumerate(display_idx) if d >= 22][0]
        ax.axvline(x=inh_start - 0.5, color='gray', linewidth=0.5)
    except (IndexError, TypeError):
        pass


# ── Panel c: Delta-correlation histograms ─────────────────────────────────────
def plot_panel_c(ax, data):
    """Histograms of correlation difference: CASCADE EXC - other algorithms."""
    algorithms = list(data['algorithms'])
    noise_levels = list(data['noise_levels'])
    nl2_idx = noise_levels.index(2) if 2 in noise_levels else 0

    # Reference: CASCADE EXC
    ref_key = 'cascade_exc_corr'
    if ref_key not in data:
        ref_key = 'cascade_tuned_corr'
    if ref_key not in data:
        ax.text(0.5, 0.5, 'No CASCADE data', ha='center', va='center',
               transform=ax.transAxes)
        return

    ref_corr = data[ref_key][nl2_idx, :]

    others = [a for a in algorithms if a not in ('cascade_tuned', 'cascade_exc')]

    n_plots = len(others)
    if n_plots == 0:
        ax.text(0.5, 0.5, 'Only CASCADE available\n(need comparison algorithms)',
               ha='center', va='center', transform=ax.transAxes,
               fontsize=7, style='italic', color='gray')
        ax.set_title('Comparison with CASCADE', fontsize=8)
        return

    # Stack histograms vertically (one per algorithm, like paper)
    for i, algo in enumerate(others):
        key = f'{algo}_corr'
        if key not in data:
            continue
        other_corr = data[key][nl2_idx, :]
        delta = ref_corr - other_corr
        valid = delta[~np.isnan(delta)]
        if len(valid) == 0:
            continue

        style = ALGO_STYLE.get(algo, {'color': 'gray', 'label': algo})
        ax.hist(valid, bins=np.linspace(-0.6, 0.3, 25),
               alpha=0.7, color=style['color'],
               edgecolor='white', linewidth=0.3,
               label=style['label'])

    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel(r'$\Delta$ correlation', fontsize=6)
    ax.set_ylabel('Histogram count (neurons)', fontsize=6)
    ax.set_title('Comparison with CASCADE', fontsize=8)
    ax.legend(fontsize=5, loc='upper left')
    ax.tick_params(labelsize=5)


# ── Panels d-f: Metric vs noise level line plots ─────────────────────────────
def _plot_vs_noise(ax, data, metric, ylabel, title, ylim=None):
    """Line plot of metric vs noise level with shaded SEM."""
    algorithms = list(data['algorithms'])
    noise_levels = data['noise_levels']

    for algo in algorithms:
        key = f'{algo}_{metric}'
        if key not in data:
            continue
        mat = data[key]  # (n_noise, n_datasets)
        if mat.ndim != 2:
            continue

        means = np.nanmean(mat, axis=1)
        n_valid = np.sum(~np.isnan(mat), axis=1)
        sems = np.nanstd(mat, axis=1) / np.sqrt(np.maximum(n_valid, 1))

        style = ALGO_STYLE.get(algo, {'color': 'gray', 'label': algo, 'ls': '-', 'lw': 1})
        ax.plot(noise_levels, means, color=style['color'], label=style['label'],
               linestyle=style['ls'], linewidth=style['lw'])
        ax.fill_between(noise_levels, means - sems, means + sems,
                        color=style['color'], alpha=0.15)

    ax.set_xlabel('Standardized noise level', fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=5, loc='best')
    ax.set_xlim(noise_levels[0], noise_levels[-1])
    if ylim:
        ax.set_ylim(ylim)
    ax.tick_params(labelsize=5)


def plot_panel_d(ax, data):
    _plot_vs_noise(ax, data, 'corr', 'Correlation with ground truth',
                   'Correlation', ylim=(0.3, 0.95))


def plot_panel_e(ax, data):
    _plot_vs_noise(ax, data, 'error', 'Relative error', 'Error')


def plot_panel_f(ax, data):
    _plot_vs_noise(ax, data, 'bias', 'Relative bias', 'Bias')
    ax.axhline(y=0, color='gray', linewidth=0.4)


# ── Panel a: Example traces ──────────────────────────────────────────────────
def plot_panel_a(ax, data):
    ax.text(0.5, 0.5,
            'Panel a: Example traces\n'
            '(dF/F + predictions)\n\n'
            'Run fig4_example_traces.py\n'
            'to generate this panel',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=7, style='italic', color='gray',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Example calcium imaging recording', fontsize=8)
    ax.axis('off')


# ── Panels g-j: Detailed comparison (placeholders) ───────────────────────────
def plot_panel_g(ax, data):
    ax.text(0.5, 0.5, 'Panel g: Spike-rate bias\nscatter plot',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=6, style='italic', color='gray')
    ax.set_xlabel('Ground truth spike rate', fontsize=6)
    ax.set_ylabel('Inferred spike rate', fontsize=6)
    ax.set_title('Spike-rate bias', fontsize=8)
    ax.tick_params(labelsize=5)


def plot_panel_h(ax, data):
    ax.text(0.5, 0.5, 'Panel h: Shared variability\n(correlation between\nalgorithm predictions)',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=6, style='italic', color='gray')
    ax.set_title('Shared variability', fontsize=8)
    ax.tick_params(labelsize=5)


def plot_panel_i(ax, data):
    ax.text(0.5, 0.5, 'Panel i: Shared errors\nhistogram',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=6, style='italic', color='gray')
    ax.set_title('Shared errors', fontsize=8)
    ax.tick_params(labelsize=5)


def plot_panel_j(ax, data):
    ax.text(0.5, 0.5, 'Panel j: Shared errors\nheatmap (FP/FN)',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=6, style='italic', color='gray')
    ax.set_title('Shared errors', fontsize=8)
    ax.tick_params(labelsize=5)


# ── Individual panels ─────────────────────────────────────────────────────────
def save_individual_panels(output_dir, data):
    os.makedirs(output_dir, exist_ok=True)

    panels = [
        ('a', plot_panel_a, (10, 5)),
        ('b', plot_panel_b, (12, 3)),
        ('c', plot_panel_c, (5, 5)),
        ('d', plot_panel_d, (5, 4)),
        ('e', plot_panel_e, (5, 4)),
        ('f', plot_panel_f, (5, 4)),
        ('g', plot_panel_g, (4, 4)),
        ('h', plot_panel_h, (3, 4)),
        ('i', plot_panel_i, (4, 4)),
        ('j', plot_panel_j, (4, 4)),
    ]

    for name, fn, figsize in panels:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fn(ax, data)
        fig.tight_layout()
        for ext in ['pdf', 'png']:
            fig.savefig(os.path.join(output_dir, f'fig4_panel_{name}.{ext}'),
                       dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved panel {name}")


# ── Assembled Figure 4 ───────────────────────────────────────────────────────
def assemble_figure4(output_dir, data):
    os.makedirs(output_dir, exist_ok=True)

    # Paper layout:
    # Row 1: [a (wide, ~3 cols)] [c (~3 cols)]
    # Row 2: [b (full width)]
    # Row 3: [d (2 cols)] [e (2 cols)] [f (2 cols)]
    # Row 4: [g (2 cols)] [h (1 col)] [i (2 cols)] [j (1 col)]

    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(4, 6, figure=fig,
                          height_ratios=[2.5, 2, 2, 2],
                          hspace=0.35, wspace=0.5)

    # Row 1
    ax_a = fig.add_subplot(gs[0, :3])
    plot_panel_a(ax_a, data)
    ax_a.text(-0.03, 1.02, 'a', transform=ax_a.transAxes,
              fontsize=14, fontweight='bold')

    ax_c = fig.add_subplot(gs[0, 3:])
    plot_panel_c(ax_c, data)
    ax_c.text(-0.05, 1.02, 'c', transform=ax_c.transAxes,
              fontsize=14, fontweight='bold')

    # Row 2: full-width heatmap
    ax_b = fig.add_subplot(gs[1, :])
    plot_panel_b(ax_b, data)
    ax_b.text(-0.02, 1.05, 'b', transform=ax_b.transAxes,
              fontsize=14, fontweight='bold')

    # Row 3: d, e, f
    ax_d = fig.add_subplot(gs[2, :2])
    plot_panel_d(ax_d, data)
    ax_d.text(-0.08, 1.02, 'd', transform=ax_d.transAxes,
              fontsize=14, fontweight='bold')

    ax_e = fig.add_subplot(gs[2, 2:4])
    plot_panel_e(ax_e, data)
    ax_e.text(-0.08, 1.02, 'e', transform=ax_e.transAxes,
              fontsize=14, fontweight='bold')

    ax_f = fig.add_subplot(gs[2, 4:])
    plot_panel_f(ax_f, data)
    ax_f.text(-0.08, 1.02, 'f', transform=ax_f.transAxes,
              fontsize=14, fontweight='bold')

    # Row 4: g, h, i, j
    ax_g = fig.add_subplot(gs[3, :2])
    plot_panel_g(ax_g, data)
    ax_g.text(-0.08, 1.02, 'g', transform=ax_g.transAxes,
              fontsize=14, fontweight='bold')

    ax_h = fig.add_subplot(gs[3, 2])
    plot_panel_h(ax_h, data)
    ax_h.text(-0.15, 1.02, 'h', transform=ax_h.transAxes,
              fontsize=14, fontweight='bold')

    ax_i = fig.add_subplot(gs[3, 3:5])
    plot_panel_i(ax_i, data)
    ax_i.text(-0.08, 1.02, 'i', transform=ax_i.transAxes,
              fontsize=14, fontweight='bold')

    ax_j = fig.add_subplot(gs[3, 5])
    plot_panel_j(ax_j, data)
    ax_j.text(-0.15, 1.02, 'j', transform=ax_j.transAxes,
              fontsize=14, fontweight='bold')

    fig.suptitle('Fig. 4 | Comparison with model-based algorithms',
                fontsize=13, fontweight='bold', y=0.995)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'Figure_4.{ext}'),
                   dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nAssembled Figure 4 saved to {output_dir}/Figure_4.{{pdf,png}}")


if __name__ == '__main__':
    data = load_results()
    output_dir = os.path.join(project_root, 'notebooks', 'figures')
    save_individual_panels(output_dir, data)
    assemble_figure4(output_dir, data)
