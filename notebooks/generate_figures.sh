#!/bin/bash
# Generate all figures for the CASCADE paper reproduction.
# Run from the CascadeTorch project root.
#
# Usage:
#   bash notebooks/generate_figures.sh           # Plot only (uses pre-computed data)
#   bash notebooks/generate_figures.sh --compute  # Compute benchmark + plot

set -e
cd "$(dirname "$0")/.."

PYTHON="/home/yz/anaconda3/envs/cascadetorch/bin/python"
export SETUPTOOLS_USE_DISTUTILS=stdlib
export PYTHONUNBUFFERED=1

echo "=== CASCADE Paper Figure Reproduction ==="
echo "Working directory: $(pwd)"
echo ""

if [[ "$1" == "--compute" ]]; then
    echo "--- Phase 1: Computing Figure 3 benchmark (this takes ~5 hours) ---"
    $PYTHON notebooks/fig3_compute_benchmark.py

    echo ""
    echo "--- Phase 2: Computing Figure 4 comparison ---"
    $PYTHON notebooks/fig4_compute_comparison.py
fi

echo ""
echo "--- Generating Figure 3 ---"
$PYTHON notebooks/fig3_plot.py

echo ""
echo "--- Generating Figure 4 example traces (panel a) ---"
$PYTHON notebooks/fig4_example_traces.py

if [ -f notebooks/fig4_comparison_results.npz ]; then
    echo ""
    echo "--- Generating Figure 4 (full) ---"
    $PYTHON notebooks/fig4_plot.py
fi

echo ""
echo "=== Done! ==="
echo "Figures saved to: notebooks/figures/"
ls -la notebooks/figures/Figure_*.{pdf,png} 2>/dev/null
