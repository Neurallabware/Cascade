#!/usr/bin/env python3
"""
Compute cross-dataset benchmark metrics for Figure 3 of the CASCADE paper.

Trains models on each dataset individually and tests on all others,
producing correlation, error, and bias matrices.

Additional rows: NAOMi simulated data, Global EXC, Global INH.

All datasets resampled to 7.5 Hz, noise level 2.

Output: notebooks/fig3_benchmark_results.npz
"""

import os
import warnings
warnings.filterwarnings('ignore')
import sys
import shutil
import time
import gc
import numpy as np
from copy import deepcopy
from scipy.ndimage import gaussian_filter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)
sys.path.insert(0, project_root)

import torch
from cascade2p import cascade, config, utils

# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLING_RATE = 7.5
NOISE_LEVEL = 2
SMOOTHING_SEC = 0.2
WINDOWSIZE = 64
BEFORE_FRAC = 0.5
GROUND_TRUTH_FOLDER = '/mnt/nas02/Dataset/CascadeTorch/Ground_truth'

# 26 datasets: DS01-DS21 (excitatory) + DS22-DS26 (inhibitory)
ALL_DATASETS = [
    'DS01-OGB1-m-V1',
    'DS02-OGB1-2-m-V1',
    'DS03-Cal520-m-S1',
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

DATASET_LABELS = [
    'OGB-1, mouse',      'OGB-1, mouse',      'Cal-520, mouse',
    'OGB-1, fish',       'Cal-520, fish',      'GCaMP6f, fish',
    'GCaMP6f, fish',     'GCaMP6f, fish',      'GCaMP6f, mouse',
    'GCaMP6f, mouse',    'GCaMP6f, mouse',     'GCaMP6s, mouse',
    'GCaMP6s, mouse',    'GCaMP6s, mouse',     'GCaMP6s, mouse',
    'GCaMP6s, mouse',    'GCaMP5k, mouse',     'R-CaMP1.07, mouse',
    'R-CaMP1.07, mouse', 'jRCaMP1a, mouse',   'jRGECO1a, mouse',
    'OGB-1, mouse SOM',  'OGB-1, mouse PV',   'GCaMP6f, mouse PV',
    'GCaMP6f, mouse SOM','GCaMP6f, mouse VIP',
]

N_DATASETS = len(ALL_DATASETS)
N_EXC = 21
N_INH = 5

NAOMI_DATASET = 'X-NAOMi-GCaMP6f-simulated'


def compute_metrics(gt, sr, smoothing_samples):
    """Compute correlation, error, bias."""
    gt = np.asarray(gt).flatten()
    sr = np.asarray(sr).flatten()
    nnan = ~np.isnan(sr) & ~np.isnan(gt)
    gt = gt[nnan]; sr = sr[nnan]
    if len(gt) == 0 or np.std(gt) == 0 or np.std(sr) == 0:
        return np.nan, np.nan, np.nan

    corr = np.corrcoef(gt, sr, rowvar=False)[0, 1]

    gt_s = gaussian_filter(gt.astype(float), sigma=smoothing_samples)
    sr_s = gaussian_filter(sr.astype(float), sigma=smoothing_samples)
    diff = sr_s - gt_s
    signal = np.sum(gt_s)
    if signal <= 0:
        return corr, np.nan, np.nan

    error = np.sum(np.abs(diff)) / signal
    bias = np.sum(diff) / signal
    return corr, error, bias


def preload_test_data():
    """Preload all 26 test datasets to avoid repeated I/O."""
    print("Preloading all test datasets...")
    test_data = {}
    for i, ds in enumerate(ALL_DATASETS):
        try:
            folder = [os.path.join(GROUND_TRUTH_FOLDER, ds)]
            calcium, gt = utils.preprocess_groundtruth_artificial_noise_balanced(
                ground_truth_folders=folder,
                before_frac=BEFORE_FRAC, windowsize=WINDOWSIZE,
                after_frac=1 - BEFORE_FRAC, noise_level=NOISE_LEVEL,
                sampling_rate=SAMPLING_RATE,
                smoothing=SMOOTHING_SEC * SAMPLING_RATE,
                omission_list=[], permute=0, verbose=0, replicas=0,
            )
            # Extract center timepoint
            test_data[ds] = {
                'calcium': calcium[:, WINDOWSIZE // 2].flatten(),
                'gt': gt.flatten(),
            }
            print(f"  [{i+1}/{N_DATASETS}] {ds}: {len(test_data[ds]['gt'])} samples")
        except Exception as e:
            print(f"  [{i+1}/{N_DATASETS}] {ds}: FAILED ({e})")
            test_data[ds] = None
    return test_data


def train_model_fast(training_datasets, model_name):
    """Train a temporary model."""
    cfg = {
        'model_name': model_name,
        'sampling_rate': SAMPLING_RATE,
        'training_datasets': training_datasets,
        'noise_levels': [NOISE_LEVEL],
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
        'batch_size': 4096,  # larger batch for faster training
        'training_finished': 'No',
        'verbose': 0,
    }
    cascade.create_model_folder(cfg)
    cascade.train_model(cfg['model_name'], ground_truth_folder=GROUND_TRUTH_FOLDER)
    return cfg['model_name']


def evaluate_model(model_name, test_data):
    """Evaluate model on all preloaded test datasets."""
    smoothing = SMOOTHING_SEC * SAMPLING_RATE
    corrs = np.full(N_DATASETS, np.nan)
    errs = np.full(N_DATASETS, np.nan)
    biases = np.full(N_DATASETS, np.nan)

    for i, ds in enumerate(ALL_DATASETS):
        if test_data[ds] is None:
            continue
        try:
            calcium = test_data[ds]['calcium']
            gt = test_data[ds]['gt']
            sr = cascade.predict(model_name, calcium.reshape(1, -1), verbosity=0)
            sr = np.squeeze(sr)
            c, e, b = compute_metrics(gt, sr, smoothing)
            corrs[i] = c; errs[i] = e; biases[i] = b
        except Exception as ex:
            print(f"    eval error on {ds}: {ex}")

    return corrs, errs, biases


def cleanup_model(model_name):
    """Remove temporary model files."""
    path = os.path.join('Pretrained_models', model_name)
    if os.path.exists(path):
        shutil.rmtree(path)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_results(corr_mat, err_mat, bias_mat, completed, partial=False):
    """Save results to disk."""
    suffix = '_partial' if partial else ''
    np.savez(
        os.path.join(project_root, 'notebooks', f'fig3_benchmark_results{suffix}.npz'),
        corr_matrix=corr_mat, error_matrix=err_mat, bias_matrix=bias_mat,
        dataset_names=ALL_DATASETS, dataset_labels=DATASET_LABELS,
        excitatory_datasets=ALL_DATASETS[:N_EXC],
        inhibitory_datasets=ALL_DATASETS[N_EXC:],
        sampling_rate=SAMPLING_RATE, noise_level=NOISE_LEVEL,
        smoothing_sec=SMOOTHING_SEC, completed_rows=completed,
    )


def main():
    print("=" * 70)
    print("CASCADE Figure 3: Cross-dataset generalization benchmark")
    print(f"Rate={SAMPLING_RATE}Hz, Noise={NOISE_LEVEL}, Datasets={N_DATASETS}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    # Rows: 0-25 single datasets, 26=NAOMi, 27=GlobalEXC, 28=GlobalINH
    n_total = N_DATASETS + 3
    corr_mat = np.full((n_total, N_DATASETS), np.nan)
    err_mat = np.full((n_total, N_DATASETS), np.nan)
    bias_mat = np.full((n_total, N_DATASETS), np.nan)

    # Phase 0: Preload test data
    test_data = preload_test_data()

    start = time.time()

    # Phase 1: Single-dataset models (rows 0-25)
    for i, ds in enumerate(ALL_DATASETS):
        t0 = time.time()
        model_name = f'temp_bench_{i:02d}'
        print(f"\n[{i+1}/{n_total}] Train on: {ds}")

        try:
            train_model_fast([ds], model_name)
            c, e, b = evaluate_model(model_name, test_data)
            corr_mat[i] = c; err_mat[i] = e; bias_mat[i] = b
            dt = time.time() - t0
            median_corr = np.nanmedian(c)
            print(f"  Done in {dt:.0f}s. Median corr={median_corr:.3f}")
        except Exception as ex:
            print(f"  FAILED: {ex}")
        finally:
            cleanup_model(model_name)

        # Save partial results every 5 datasets
        if (i + 1) % 5 == 0:
            save_results(corr_mat, err_mat, bias_mat, i + 1, partial=True)
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (n_total - i - 1)
            print(f"  [{i+1}/{n_total}] Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")

    # Phase 2: NAOMi model (row 26)
    print(f"\n[NAOMi] Train on: {NAOMI_DATASET}")
    try:
        train_model_fast([NAOMI_DATASET], 'temp_bench_naomi')
        c, e, b = evaluate_model('temp_bench_naomi', test_data)
        corr_mat[N_DATASETS] = c; err_mat[N_DATASETS] = e; bias_mat[N_DATASETS] = b
    except Exception as ex:
        print(f"  FAILED: {ex}")
    finally:
        cleanup_model('temp_bench_naomi')

    # Phase 3: Global EXC model (row 27) — all exc datasets except DS01
    print(f"\n[Global EXC] Train on: all excitatory except DS01")
    exc_train = ALL_DATASETS[1:N_EXC]
    try:
        train_model_fast(exc_train, 'temp_bench_exc')
        c, e, b = evaluate_model('temp_bench_exc', test_data)
        corr_mat[N_DATASETS + 1] = c; err_mat[N_DATASETS + 1] = e; bias_mat[N_DATASETS + 1] = b
    except Exception as ex:
        print(f"  FAILED: {ex}")
    finally:
        cleanup_model('temp_bench_exc')

    # Phase 4: Global INH model (row 28)
    print(f"\n[Global INH] Train on: all inhibitory datasets")
    inh_train = ALL_DATASETS[N_EXC:]
    try:
        train_model_fast(inh_train, 'temp_bench_inh')
        c, e, b = evaluate_model('temp_bench_inh', test_data)
        corr_mat[N_DATASETS + 2] = c; err_mat[N_DATASETS + 2] = e; bias_mat[N_DATASETS + 2] = b
    except Exception as ex:
        print(f"  FAILED: {ex}")
    finally:
        cleanup_model('temp_bench_inh')

    # Save final
    save_results(corr_mat, err_mat, bias_mat, n_total, partial=False)
    total = time.time() - start
    print(f"\n{'='*70}")
    print(f"DONE! Total: {total/60:.1f} min ({total/3600:.1f} hours)")
    print(f"Results: notebooks/fig3_benchmark_results.npz")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
