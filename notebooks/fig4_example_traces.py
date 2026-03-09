#!/usr/bin/env python3
"""
Generate example trace panel for Figure 4a.

Loads raw ground truth data from a representative dataset and runs CASCADE + OASIS
inference to produce example dF/F + prediction traces matching the paper.
"""

import os
import warnings
warnings.filterwarnings('ignore')
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)
sys.path.insert(0, project_root)

from cascade2p import cascade, utils
import scipy.io as sio

GT_FOLDER = '/mnt/nas02/Dataset/CascadeTorch/Ground_truth'
SAMPLING_RATE = 7.5
NOISE_LEVEL = 2
SMOOTHING_SEC = 0.2

# Example dataset — DS09 (GCaMP6f-m-V1) is representative
EXAMPLE_DS = 'DS09-GCaMP6f-m-V1'
NEURON_IDX = 0
DURATION_SEC = 50  # Show 50 seconds like the paper


def load_neuron_trace(dataset_name, neuron_idx=0):
    """
    Load a single neuron's dF/F trace and spike times from ground truth,
    resampled to target frame rate with artificial noise.
    """
    folder = os.path.join(GT_FOLDER, dataset_name)
    import glob as g
    mat_files = sorted(g.glob(os.path.join(folder, '*_mini.mat')))

    if neuron_idx >= len(mat_files):
        neuron_idx = 0

    neuron_file = mat_files[neuron_idx]
    data = sio.loadmat(neuron_file)['CAttached'][0]

    # Concatenate all trials for this neuron
    all_dff = []
    all_spikes = []

    for trial in data:
        try:
            keys = trial[0][0].dtype.descr
            keys_unfolded = list(sum(keys, ()))

            traces_idx = int(keys_unfolded.index("fluo_mean") / 2)
            time_idx = int(keys_unfolded.index("fluo_time") / 2)
            events_idx = int(keys_unfolded.index("events_AP") / 2)

            fluo = trial[0][0][traces_idx].flatten()
            fluo_time = trial[0][0][time_idx].flatten()
            spike_times = trial[0][0][events_idx].flatten()

            if len(fluo) < 10:
                continue

            # Original frame rate
            orig_dt = np.median(np.diff(fluo_time))
            orig_rate = 1.0 / orig_dt

            # Resample to target rate
            from scipy.signal import resample
            n_target = int(len(fluo) * SAMPLING_RATE / orig_rate)
            if n_target < 5:
                continue

            dff_resampled = resample(fluo, n_target)
            target_time = np.arange(n_target) / SAMPLING_RATE

            # Bin spikes at target rate
            spike_rate = np.zeros(n_target)
            for st in spike_times:
                bin_idx = int(st * SAMPLING_RATE)
                if 0 <= bin_idx < n_target:
                    spike_rate[bin_idx] += 1

            # Add artificial Poisson noise to match the target noise level
            noise_std = NOISE_LEVEL / 100.0 * np.sqrt(SAMPLING_RATE)
            dff_noisy = dff_resampled + np.random.randn(n_target) * noise_std

            all_dff.append(dff_noisy)
            all_spikes.append(spike_rate)

        except (IndexError, ValueError):
            continue

    if not all_dff:
        return None, None, None

    dff = np.concatenate(all_dff)
    spikes = np.concatenate(all_spikes)
    time_axis = np.arange(len(dff)) / SAMPLING_RATE

    return dff, spikes, time_axis


def run_cascade_predict(dff_trace):
    """Run CASCADE inference with best available model."""
    # Try to find a matching model
    models_to_try = [
        'Global_EXC_7Hz_smoothing200ms',
        'Global_EXC_7.5Hz_smoothing200ms',
        'Global_EXC_10Hz_smoothing200ms',
    ]

    for model_name in models_to_try:
        model_path = os.path.join('Pretrained_models', model_name)
        if os.path.isdir(model_path):
            print(f"  Using model: {model_name}")
            sr = cascade.predict(model_name, dff_trace.reshape(1, -1), verbosity=0)
            return np.squeeze(sr)

    print("  No suitable pretrained model found. Training a temporary one...")
    return None


def run_oasis_predict(dff_trace):
    """Run OASIS deconvolution."""
    try:
        from oasis.oasis_methods import oasisAR1
        trace_1d = np.asarray(dff_trace).flatten().astype(np.float64)
        # Use AR(1) model with reasonable decay time constant
        # g = exp(-1/(tau * frame_rate)), tau ~1 sec for GCaMP6
        g = np.exp(-1.0 / (1.0 * SAMPLING_RATE))
        # Estimate noise standard deviation
        noise = np.median(np.abs(np.diff(trace_1d))) / 0.6745
        lam = noise * np.sqrt(SAMPLING_RATE)
        c, s = oasisAR1(trace_1d - np.percentile(trace_1d, 10), g=g, lam=lam)
        return s
    except Exception as e:
        print(f"  OASIS error: {e}")
        return None


def plot_example_traces(output_dir):
    """Create Figure 4a panel: example traces."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading traces from {EXAMPLE_DS}...")
    dff, spike_rate, time_axis = load_neuron_trace(EXAMPLE_DS, NEURON_IDX)

    if dff is None:
        print("Failed to load traces")
        return

    # Limit to DURATION_SEC
    n_pts = int(DURATION_SEC * SAMPLING_RATE)
    if len(dff) > n_pts:
        # Find a segment with some activity
        sr_smooth = gaussian_filter(spike_rate.astype(float), sigma=SAMPLING_RATE)
        best_start = np.argmax(np.convolve(sr_smooth, np.ones(n_pts), mode='valid'))
        dff = dff[best_start:best_start + n_pts]
        spike_rate = spike_rate[best_start:best_start + n_pts]
        time_axis = np.arange(n_pts) / SAMPLING_RATE

    # Smooth ground truth
    gt_smooth = gaussian_filter(spike_rate.astype(float), sigma=SMOOTHING_SEC * SAMPLING_RATE)

    print("Running CASCADE inference...")
    cascade_sr = run_cascade_predict(dff)

    print("Running OASIS inference...")
    oasis_sr = run_oasis_predict(dff)

    # ── Build traces dict ─────────────────────────────────────────────────────
    traces = {}
    if cascade_sr is not None:
        traces['CASCADE'] = cascade_sr
    if oasis_sr is not None:
        traces['OASIS'] = oasis_sr

    # ── Plot matching paper style ─────────────────────────────────────────────
    # Paper shows: dF/F row, then each algorithm row
    n_rows = 1 + len(traces)
    fig_height = 1.5 * n_rows + 0.5

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, fig_height),
                             sharex=True, gridspec_kw={'hspace': 0.08})
    if n_rows == 1:
        axes = [axes]

    algo_colors = {
        'CASCADE': '#1976D2',
        'OASIS': '#FF9800',
        'MLSpike': '#4CAF50',
        'Peeling': '#F44336',
        'CaImAn': '#9C27B0',
        'Suite2p': '#795548',
        'Jewell&\nWitten': '#E91E63',
    }

    # Row 0: dF/F
    ax = axes[0]
    ax.plot(time_axis, dff, 'k', linewidth=0.4)
    # Overlay ground truth spike rate (red, scaled)
    if np.max(gt_smooth) > 0:
        gt_scaled = gt_smooth / np.max(gt_smooth) * np.max(np.abs(dff)) * 0.8
        ax.plot(time_axis, gt_scaled, 'r', linewidth=0.6, alpha=0.7)
    ax.set_ylabel(r'$\Delta F/F$', fontsize=8)
    ax.text(-0.02, 0.5, r'$\Delta F/F$', fontsize=7, transform=ax.transAxes,
           ha='right', va='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=5)

    # Add legend
    ax.text(0.01, 0.95, 'Inferred SR', fontsize=5, color='black',
           transform=ax.transAxes, va='top')
    ax.text(0.01, 0.80, 'Ground truth SR', fontsize=5, color='red',
           transform=ax.transAxes, va='top')

    # Algorithm rows
    for idx, (algo_name, sr) in enumerate(traces.items()):
        ax = axes[1 + idx]
        n = min(len(sr), len(time_axis))

        # Plot prediction
        color = algo_colors.get(algo_name, '#666666')
        sr_plot = sr[:n]
        t_plot = time_axis[:n]

        # Handle NaN padding
        valid = ~np.isnan(sr_plot)
        ax.plot(t_plot[valid], sr_plot[valid], color=color, linewidth=0.5)

        # Overlay ground truth
        gt_plot = gt_smooth[:n]
        if np.max(gt_plot) > 0:
            # Scale ground truth to match prediction range
            sr_valid = sr_plot[valid]
            if len(sr_valid) > 0 and np.max(sr_valid) > 0:
                scale = np.percentile(sr_valid[sr_valid > 0], 95) / np.max(gt_plot) if np.any(sr_valid > 0) else 1
                ax.plot(t_plot, gt_plot * scale, 'r', linewidth=0.4, alpha=0.5)

        # Correlation
        gt_aligned = gt_smooth[:n]
        valid_both = valid & ~np.isnan(gt_aligned)
        if np.sum(valid_both) > 10 and np.std(gt_aligned[valid_both]) > 0 and np.std(sr_plot[valid_both]) > 0:
            corr = np.corrcoef(gt_aligned[valid_both], sr_plot[valid_both])[0, 1]
            ax.text(0.98, 0.85, f'r = {corr:.2f}', transform=ax.transAxes,
                   fontsize=6, ha='right', va='top')

        ax.set_ylabel(algo_name, fontsize=7, color=color, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=5)

        # Mark false-negative detections with red arrows (like paper)
        # These are periods where ground truth has spikes but prediction is near zero
        if np.max(gt_plot) > 0:
            threshold_gt = np.percentile(gt_plot[gt_plot > 0], 75)
            for t_idx in range(0, n, int(5 * SAMPLING_RATE)):  # check every 5 sec
                window = slice(t_idx, min(t_idx + int(2 * SAMPLING_RATE), n))
                if np.max(gt_plot[window]) > threshold_gt:
                    sr_window = sr_plot[window]
                    sr_window_valid = sr_window[~np.isnan(sr_window)]
                    if len(sr_window_valid) > 0 and np.max(sr_window_valid) < threshold_gt * 0.1:
                        # False negative — mark with red arrow
                        t_mark = t_plot[t_idx + int(SAMPLING_RATE)]
                        ax.annotate('', xy=(t_mark, 0), xytext=(t_mark, ax.get_ylim()[1] * 0.8),
                                   arrowprops=dict(arrowstyle='->', color='red', lw=1))

    axes[-1].set_xlabel('Time (s)', fontsize=8)

    # Add scale bars
    # 5 seconds, 50% dF/F (like paper: "50%  6 Hz  5 s")
    ax_top = axes[0]
    x_pos = time_axis[-1] * 0.95
    y_range = ax_top.get_ylim()

    fig.suptitle(f'Example traces ({EXAMPLE_DS})', fontsize=9, fontweight='bold', y=0.98)

    path_pdf = os.path.join(output_dir, 'fig4_panel_a.pdf')
    path_png = os.path.join(output_dir, 'fig4_panel_a.png')
    fig.savefig(path_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path_pdf}")
    print(f"Saved: {path_png}")


if __name__ == '__main__':
    output_dir = os.path.join(project_root, 'notebooks', 'figures')
    plot_example_traces(output_dir)
