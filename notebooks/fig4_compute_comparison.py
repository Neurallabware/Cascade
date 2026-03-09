#!/usr/bin/env python3
"""
Compute comparison metrics for Figure 4 of the CASCADE paper.

Compares CASCADE with model-based algorithms:
  - CASCADE (tuned per dataset + global EXC model)
  - OASIS (fast online deconvolution, Python)
  - MLSpike, Peeling, CaImAn, Suite2p, Jewell&Witten (if available)

Figure 4 panels:
  a) Example traces (dF/F + predictions from all algorithms)
  b) Correlation heatmap across datasets for all algorithms (noise level 2)
  c) Histograms comparing CASCADE vs each algorithm
  d) Correlation vs noise level
  e) Error vs noise level
  f) Bias vs noise level
  g) Spike-rate bias scatter
  h) Shared variability dot plot
  i) Shared errors histogram
  j) Shared errors heatmap

Output: notebooks/fig4_comparison_results.npz
"""

import os
import warnings
warnings.filterwarnings('ignore')
import sys
import shutil
import time
import numpy as np
from copy import deepcopy
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve

# Ensure project root is on path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)
sys.path.insert(0, project_root)

from cascade2p import cascade, config, utils

# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLING_RATE = 7.5
NOISE_LEVELS = list(range(2, 10))  # noise levels 2-9
SMOOTHING_SEC = 0.2
WINDOWSIZE = 64
BEFORE_FRAC = 0.5
GROUND_TRUTH_FOLDER = '/mnt/nas02/Dataset/CascadeTorch/Ground_truth'

# 26 datasets (matching paper: DS03 omitted from Fig4 because recordings <10s)
# Paper says: "DS #03 was omitted for all comparisons in Fig. 4 because
# the short recordings (<10 s) could not be processed by all algorithms"
ALL_DATASETS = [
    'DS01-OGB1-m-V1',
    'DS02-OGB1-2-m-V1',
    # DS03 omitted
    'DS04-OGB1-zf-pDp',
    'DS05-Cal520-zf-pDp',
    'DS06-GCaMP6f-zf-aDp',
    'DS07-GCaMP6f-zf-dD',
    'DS08-GCaMP6f-zf-OB',
    'DS09-GCaMP6f-m-V1',
    'DS10-GCaMP6f-m-V1-neuropil-corrected',
    'DS11-GCaMP6f-m-V1-neuropil-corrected',
    'DS12-GCaMP6s-m-V1-neuropil-corrected',
    'DS13-GCaMP6s-m-V1-neuropil-corrected',
    'DS14-GCaMP6s-m-V1',
    'DS15-GCaMP6s-m-V1',
    'DS16-GCaMP6s-m-V1',
    'DS17-GCaMP5k-m-V1',
    'DS18-R-CaMP-m-CA3',
    'DS19-R-CaMP-m-S1',
    'DS20-jRCaMP1a-m-V1',
    'DS21-jGECO1a-m-V1',
    'DS22-OGB1-m-SST-V1',
    'DS23-OGB1-m-PV-V1',
    'DS24-GCaMP6f-m-PV-V1',
    'DS25-GCaMP6f-m-SST-V1',
    'DS26-GCaMP6f-m-VIP-V1',
]
N_DATASETS = len(ALL_DATASETS)

# Display indices (matching paper's Fig 4b x-axis numbering)
DISPLAY_INDICES = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]


# ── Algorithm implementations ─────────────────────────────────────────────────

def run_oasis_deconv(calcium_trace, frame_rate):
    """Run OASIS deconvolution on a single trace."""
    try:
        from oasis.oasis_methods import oasisAR1
        trace_1d = np.asarray(calcium_trace).flatten().astype(np.float64)
        # AR(1) with GCaMP6 decay constant (~1 sec)
        g = np.exp(-1.0 / (1.0 * frame_rate))
        # Estimate noise from high-frequency fluctuations
        noise = np.median(np.abs(np.diff(trace_1d))) / 0.6745
        lam = noise * np.sqrt(frame_rate)
        # Subtract baseline
        baseline = np.percentile(trace_1d, 10)
        c, s = oasisAR1(trace_1d - baseline, g=g, lam=lam)
        return s
    except ImportError:
        return None
    except Exception as e:
        print(f"  OASIS error: {e}")
        return None


def compute_metrics(ground_truth, spike_rates, smoothing_samples):
    """Compute correlation, error, and bias."""
    gt = np.asarray(ground_truth).flatten()
    sr = np.asarray(spike_rates).flatten()

    nnan_ix = ~np.isnan(sr) & ~np.isnan(gt)
    gt = gt[nnan_ix]
    sr = sr[nnan_ix]

    if len(gt) == 0 or np.std(gt) == 0 or np.std(sr) == 0:
        return np.nan, np.nan, np.nan

    gt_smooth = gaussian_filter(gt.astype(float), sigma=smoothing_samples)
    sr_smooth = gaussian_filter(sr.astype(float), sigma=smoothing_samples)

    corr = np.corrcoef(gt, sr, rowvar=False)[0, 1]

    error_diff = sr_smooth - gt_smooth
    error_pos = np.sum(error_diff[error_diff > 0])
    error_neg = np.sum(error_diff[error_diff < 0])
    error_total = np.sum(np.abs(error_diff))
    signal = np.sum(gt_smooth)

    if signal > 0:
        error = error_total / signal
        bias = (error_pos + error_neg) / signal
    else:
        error = np.nan
        bias = np.nan

    return corr, error, bias


def get_test_data(dataset_name, noise_level):
    """Load and preprocess test data for a dataset at a given noise level."""
    test_folder = [os.path.join(GROUND_TRUTH_FOLDER, dataset_name)]

    calcium, ground_truth = utils.preprocess_groundtruth_artificial_noise_balanced(
        ground_truth_folders=test_folder,
        before_frac=BEFORE_FRAC,
        windowsize=WINDOWSIZE,
        after_frac=1 - BEFORE_FRAC,
        noise_level=noise_level,
        sampling_rate=SAMPLING_RATE,
        smoothing=SMOOTHING_SEC * SAMPLING_RATE,
        omission_list=[],
        permute=0,
        verbose=0,
        replicas=0,
    )
    calcium_center = calcium[:, WINDOWSIZE // 2].flatten()
    ground_truth_flat = ground_truth.flatten()
    return calcium_center, ground_truth_flat


# ── Phase 1: CASCADE tuned models (leave-one-out) ────────────────────────────

def run_cascade_tuned(noise_level=2):
    """Train tuned CASCADE models (leave-one-out) and evaluate."""
    print(f"\n=== CASCADE Tuned (noise level {noise_level}) ===")
    correlations = np.full(N_DATASETS, np.nan)
    errors = np.full(N_DATASETS, np.nan)
    biases = np.full(N_DATASETS, np.nan)
    smoothing_samples = SMOOTHING_SEC * SAMPLING_RATE

    for i, test_ds in enumerate(ALL_DATASETS):
        train_ds = [d for d in ALL_DATASETS if d != test_ds]
        model_name = f'temp_fig4_cascade_tuned_{i:02d}'

        cfg = {
            'model_name': model_name,
            'sampling_rate': SAMPLING_RATE,
            'training_datasets': train_ds,
            'noise_levels': [noise_level],
            'smoothing': SMOOTHING_SEC,
            'causal_kernel': 0,
            'windowsize': WINDOWSIZE,
            'before_frac': BEFORE_FRAC,
            'filter_sizes': [31, 19, 5],
            'filter_numbers': [30, 40, 50],
            'dense_expansion': 10,
            'loss_function': 'mean_squared_error',
            'optimizer': 'Adagrad',
            'nr_of_epochs': 20,
            'ensemble_size': 5,
            'batch_size': 1024,
            'training_finished': 'No',
            'verbose': 0,
        }

        try:
            print(f"  [{i+1}/{N_DATASETS}] Training tuned model, testing on {test_ds}...")
            cascade.create_model_folder(cfg)
            cascade.train_model(cfg['model_name'], ground_truth_folder=GROUND_TRUTH_FOLDER)

            calcium, gt = get_test_data(test_ds, noise_level)
            sr = cascade.predict(cfg['model_name'], calcium.reshape(1, -1), verbosity=0)
            sr = np.squeeze(sr)

            corr, err, bias = compute_metrics(gt, sr, smoothing_samples)
            correlations[i] = corr
            errors[i] = err
            biases[i] = bias
            print(f"    corr={corr:.4f}, err={err:.4f}, bias={bias:.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
        finally:
            model_path = os.path.join('Pretrained_models', model_name)
            if os.path.exists(model_path):
                shutil.rmtree(model_path)

    return correlations, errors, biases


def run_cascade_exc(noise_level=2):
    """Run Global EXC CASCADE model (trained on all excitatory except DS01)."""
    print(f"\n=== CASCADE EXC (noise level {noise_level}) ===")
    exc_datasets = [d for d in ALL_DATASETS if int(d[2:4]) <= 21]
    exc_train = exc_datasets[1:]  # exclude DS01

    model_name = f'temp_fig4_cascade_exc_nl{noise_level}'
    cfg = {
        'model_name': model_name,
        'sampling_rate': SAMPLING_RATE,
        'training_datasets': exc_train,
        'noise_levels': [noise_level],
        'smoothing': SMOOTHING_SEC,
        'causal_kernel': 0,
        'windowsize': WINDOWSIZE,
        'before_frac': BEFORE_FRAC,
        'filter_sizes': [31, 19, 5],
        'filter_numbers': [30, 40, 50],
        'dense_expansion': 10,
        'loss_function': 'mean_squared_error',
        'optimizer': 'Adagrad',
        'nr_of_epochs': 20,
        'ensemble_size': 5,
        'batch_size': 1024,
        'training_finished': 'No',
        'verbose': 0,
    }

    correlations = np.full(N_DATASETS, np.nan)
    errors = np.full(N_DATASETS, np.nan)
    biases = np.full(N_DATASETS, np.nan)
    smoothing_samples = SMOOTHING_SEC * SAMPLING_RATE

    try:
        cascade.create_model_folder(cfg)
        cascade.train_model(cfg['model_name'], ground_truth_folder=GROUND_TRUTH_FOLDER)

        for i, test_ds in enumerate(ALL_DATASETS):
            try:
                calcium, gt = get_test_data(test_ds, noise_level)
                sr = cascade.predict(cfg['model_name'], calcium.reshape(1, -1), verbosity=0)
                sr = np.squeeze(sr)

                corr, err, bias = compute_metrics(gt, sr, smoothing_samples)
                correlations[i] = corr
                errors[i] = err
                biases[i] = bias
            except Exception as e:
                print(f"  Error on {test_ds}: {e}")
    except Exception as e:
        print(f"  Training error: {e}")
    finally:
        model_path = os.path.join('Pretrained_models', model_name)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

    return correlations, errors, biases


def run_oasis_comparison(noise_level=2):
    """Run OASIS deconvolution on all datasets."""
    print(f"\n=== OASIS (noise level {noise_level}) ===")
    correlations = np.full(N_DATASETS, np.nan)
    errors = np.full(N_DATASETS, np.nan)
    biases = np.full(N_DATASETS, np.nan)
    smoothing_samples = SMOOTHING_SEC * SAMPLING_RATE

    for i, ds in enumerate(ALL_DATASETS):
        try:
            calcium, gt = get_test_data(ds, noise_level)

            # Run OASIS on the center-extracted trace
            sr = run_oasis_deconv(calcium, SAMPLING_RATE)
            if sr is None:
                continue

            # Smooth the OASIS output to match CASCADE's temporal resolution
            # OASIS outputs discrete spikes; convolve with Gaussian to get spike rates
            sr_smooth_for_comparison = gaussian_filter(sr.astype(float), sigma=smoothing_samples)

            # Scale to match ground truth range (OASIS outputs are in arbitrary units)
            if np.max(sr_smooth_for_comparison) > 0:
                scale = np.sum(gt) / np.sum(sr_smooth_for_comparison)
                sr_scaled = sr * scale
            else:
                sr_scaled = sr

            corr, err, bias = compute_metrics(gt, sr_scaled, smoothing_samples)
            correlations[i] = corr
            errors[i] = err
            biases[i] = bias
            print(f"  [{i+1}/{N_DATASETS}] {ds}: corr={corr:.4f}")
        except Exception as e:
            print(f"  [{i+1}/{N_DATASETS}] {ds}: ERROR {e}")

    return correlations, errors, biases


def main():
    print("=" * 70)
    print("CASCADE Figure 4: Comparison with model-based algorithms")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    print(f"Datasets: {N_DATASETS}")
    print("=" * 70)

    start_time = time.time()

    results = {}

    # ── Phase 1: CASCADE tuned (noise level 2 for Fig 4b heatmap) ──
    c, e, b = run_cascade_tuned(noise_level=2)
    results['cascade_tuned_nl2'] = {'corr': c, 'error': e, 'bias': b}

    # ── Phase 2: CASCADE EXC at multiple noise levels (Fig 4d-f) ──
    for nl in NOISE_LEVELS:
        c, e, b = run_cascade_exc(noise_level=nl)
        results[f'cascade_exc_nl{nl}'] = {'corr': c, 'error': e, 'bias': b}

    # ── Phase 3: OASIS at multiple noise levels ──
    for nl in NOISE_LEVELS:
        c, e, b = run_oasis_comparison(noise_level=nl)
        results[f'oasis_nl{nl}'] = {'corr': c, 'error': e, 'bias': b}

    # ── Save results ──
    output_path = os.path.join(project_root, 'notebooks', 'fig4_comparison_results.npz')

    # Restructure for easy access
    algorithms = ['cascade_tuned', 'cascade_exc', 'oasis']
    noise_levels_arr = np.array(NOISE_LEVELS)

    # Build per-algorithm, per-noise-level arrays
    save_dict = {
        'dataset_names': ALL_DATASETS,
        'display_indices': DISPLAY_INDICES,
        'noise_levels': noise_levels_arr,
        'algorithms': algorithms,
    }

    for algo in algorithms:
        for metric in ['corr', 'error', 'bias']:
            # Shape: (n_noise_levels, n_datasets)
            mat = np.full((len(NOISE_LEVELS), N_DATASETS), np.nan)
            for ni, nl in enumerate(NOISE_LEVELS):
                key = f'{algo}_nl{nl}'
                if key in results:
                    mat[ni, :] = results[key][metric]
            save_dict[f'{algo}_{metric}'] = mat

    np.savez(output_path, **save_dict)

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Figure 4 computation complete! Total time: {total_time/60:.1f} min")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
