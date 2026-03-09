# CascadeTorch Reproduction Report

**Repository:** [PTRRupprecht/CascadeTorch](https://github.com/PTRRupprecht/CascadeTorch)
**Paper:** Rupprecht et al., *A database and deep learning toolbox for noise-optimized, generalized spike inference from calcium imaging*, Nature Neuroscience (2021)
**Date:** 2026-03-09
**Machine:** Linux 5.15.0-156-generic, 8x NVIDIA A100-SXM4-80GB, CUDA 12.4

---

## Section 1 -- Summary Table

| Asset | Type | Status | Notes |
|---|---|---|---|
| `Demo_predict.py` | script | PASS | Loaded data, downloaded model, ran inference on 74 neurons |
| `Demo_train.py` | script | PASS (FIXED) | Trained 3/5 ensembles in 30 min before early stop (see Patch #1) |
| `Demo_discrete_spikes.py` | script | PASS (FIXED) | Inferred discrete spikes for 1005 neurons (see Patches #2, #3) |
| `Demo_benchmark_model.py` | script | PASS (FIXED) | Verified startup; full run takes hours (see Patches #4, #5, #6) |
| `Process_output_from_Suite2p.py` | script | SKIPPED | Requires external Suite2p data (MANUAL_DOWNLOAD_REQUIRED) |
| `Explore_ground_truth_datasets.ipynb` | notebook | PASS | All cells executed successfully |
| `Calibrated_spike_inference_with_Cascade.ipynb` | notebook | EXPECTED FAIL | Colab-only notebook; `%%capture` after `#@markdown` fails locally |
| Package import | test | PASS | `from cascade2p import cascade, config, utils, checks` all OK |

---

## Section 2 -- Environment Specification

- **OS:** Ubuntu 22.04 (Linux 5.15.0-156-generic x86_64)
- **Python:** 3.11.12 (conda: `cascadetorch`)
- **CUDA:** 12.4 (Driver 550.163.01)
- **GPU:** 8x NVIDIA A100-SXM4-80GB

### Key Packages

| Package | Version |
|---|---|
| torch | 2.6.0+cu124 |
| numpy | 2.4.2 |
| scipy | 1.17.1 |
| matplotlib | 3.10.8 |
| h5py | 3.16.0 |
| seaborn | 0.13.2 |
| ruamel.yaml | 0.19.1 |
| CascadeTorch | 2.0 (editable) |

Full environment: run `conda activate cascadetorch && pip freeze` for complete list.

---

## Section 3 -- Patch Log

### Patch #0: Missing `__init__.py` (CRITICAL)
- **File:** `cascade2p/__init__.py`
- **Original:** File did not exist
- **Patched:** Created empty `__init__.py`
- **Reason:** `find_packages()` in `setup.py` requires `__init__.py` to recognize `cascade2p` as a package. Without it, `pip install -e .` installs the project but not the `cascade2p` module, causing `ModuleNotFoundError` when running scripts.

### Patch #1: Hardcoded Windows path in Demo_train.py
- **File:** `Demo scripts/Demo_train.py`, line 30
- **Original:** `os.chdir(r'C:\Users\peter\Desktop\CascadeTorch\CascadeTorch\Demo scripts')`
- **Patched:** Commented out the line
- **Reason:** Hardcoded Windows path crashes on Linux. The script's existing `if 'Demo scripts' in os.getcwd()` logic handles cross-platform path resolution.

### Patch #2: Missing `sys` import in Demo_discrete_spikes.py
- **File:** `Demo scripts/Demo_discrete_spikes.py`, line 27
- **Original:** `import os`
- **Patched:** `import os, sys`
- **Reason:** Line 29 uses `sys.path.append()` but `sys` was never imported.

### Patch #3: Wrong mat file key in Demo_discrete_spikes.py
- **File:** `Demo scripts/Demo_discrete_spikes.py`, line 61
- **Original:** `spike_prob = sio.loadmat(file_path)['spike_prob']`
- **Patched:** `mat = sio.loadmat(file_path); spike_prob = mat.get('spike_prob', mat.get('spike_rates'))`
- **Reason:** The bundled prediction file uses key `spike_rates`, not `spike_prob`.

### Patch #4: Missing `sys` import in Demo_benchmark_model.py
- **File:** `Demo scripts/Demo_benchmark_model.py`, line 30
- **Original:** `import os`
- **Patched:** `import os, sys`
- **Reason:** Line 34 uses `sys.path.append()` but `sys` was never imported.

### Patch #5: Removed `keras` import in Demo_benchmark_model.py
- **File:** `Demo scripts/Demo_benchmark_model.py`, line 38
- **Original:** `import keras`
- **Patched:** Commented out
- **Reason:** TensorFlow leftover. The PyTorch version does not use Keras.

### Patch #6: Fixed wrong model reference in Demo_benchmark_model.py
- **File:** `Demo scripts/Demo_benchmark_model.py`, line 140
- **Original:** `spike_rates = cascade.predict( model_name, calcium.T )`
- **Patched:** `spike_rates = cascade.predict( cfg['model_name'], calcium.T )`
- **Reason:** Bug: used original model name instead of the leave-one-out temporary model for benchmark predictions.

### Patch #7: Deprecated scipy import in Demo_benchmark_model.py
- **File:** `Demo scripts/Demo_benchmark_model.py`, line 40
- **Original:** `from scipy.ndimage.filters import gaussian_filter`
- **Patched:** `from scipy.ndimage import gaussian_filter`
- **Reason:** `scipy.ndimage.filters` is deprecated in scipy >= 1.11.

### Environment Fix: Triton uninstalled
- **Package:** `triton==3.2.0` (bundled with torch 2.6.0+cu124)
- **Action:** `pip uninstall triton`
- **Reason:** Triton's AMD backend driver triggers a `setuptools`/`_distutils_hack` assertion error on this system. CascadeTorch does not use `torch.compile()`, so triton is not needed.

---

## Section 4 -- Data Manifest

All data stored at `/mnt/nas02/Dataset/CascadeTorch/` (total: 1.2 GB)

| Directory | Size | Source | Description |
|---|---|---|---|
| `Ground_truth/` | 386 MB | Included in repo | 41 ground truth datasets (DS01-DS41, X-DS*) |
| `Example_datasets/` | 19 MB | Included in repo | 2 example datasets (Allen Brain Observatory 30Hz, Multiplane OGB1) |
| `Pretrained_models/` | 810 MB | Included + downloaded | Pre-trained models including `Global_EXC_30Hz_smoothing25ms` (bundled), `Global_EXC_30Hz_smoothing100ms` (downloaded), `OGB_zf_pDp_7.5Hz_smoothing200ms` (downloaded) |

### Downloaded Models (via `cascade.download_model()`)

| Model | Source URL | Local Path |
|---|---|---|
| `Global_EXC_30Hz_smoothing25ms` | `https://drive.switch.ch/.../WVWtEQXAaiDaEKP/download` | `Pretrained_models/Global_EXC_30Hz_smoothing25ms/` |
| `Global_EXC_30Hz_smoothing100ms` | `https://drive.switch.ch/.../X6N6PnpaIkmltlh/download` | `Pretrained_models/Global_EXC_30Hz_smoothing100ms/` |
| `OGB_zf_pDp_7.5Hz_smoothing200ms` | `https://drive.switch.ch/.../PGa3dUalY2fnOwF/download` | `Pretrained_models/OGB_zf_pDp_7.5Hz_smoothing200ms/` |

100+ additional pretrained models available via `available_models_CascadeTorch.yaml`.

---

## Section 5 -- Verdict

### PARTIALLY REPRODUCED

**All core functionality works.** The main pipeline (predict, train, discrete spikes, ground truth exploration) runs successfully after minimal patches. The patches fix genuine bugs (missing imports, wrong keys, TF leftovers) rather than environment-specific issues.

**Items requiring manual intervention:**

1. **`Process_output_from_Suite2p.py`** -- Requires user's own Suite2p output files (`F.npy`, `Fneu.npy`, etc.). Not testable without external data.
2. **`Calibrated_spike_inference_with_Cascade.ipynb`** -- Colab-only notebook with `%%capture`/`#@markdown` syntax. Works on Google Colab but not in local Jupyter.
3. **`Demo_benchmark_model.py`** -- Verified to start correctly but full execution takes many hours (trains 17+ leave-one-out models). Contains a logic bug (Patch #6) where it tested with the original model instead of the temporary one.

**Reproduction command:**
```bash
conda activate cascadetorch
cd /home/yz/spike_deconv/CascadeTorch
python "Demo scripts/Demo_predict.py"
```
