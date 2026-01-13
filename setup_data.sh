#!/bin/bash
# =============================================================================
# Setup Script for PuckerFlow
# =============================================================================
# This script:
#   1. Extracts the compressed safe data archives (.tar.gz)
#   2. Converts the safe data formats back to pickle files
#
# Run this once after cloning the repository.
#
# Usage:
#   ./setup_data.sh
#
# Requirements:
#   - Python 3.8+
#   - RDKit
#   - NumPy
#   - PyTorch (optional, for torch tensor support)
#   - PyTorch Geometric (optional, for PyG data support)
# =============================================================================

set -e

echo "=============================================================================="
echo "PuckerFlow Data Setup"
echo "=============================================================================="
echo ""
echo "This script will:"
echo "  1. Extract compressed safe data archives"
echo "  2. Convert the safe data formats to pickle files"
echo ""
echo "This is required before running training, generation, or evaluation."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    exit 1
fi

# Check if required packages are available
echo "Checking dependencies..."
python -c "import numpy" 2>/dev/null || { echo "ERROR: numpy is not installed"; exit 1; }
python -c "from rdkit import Chem" 2>/dev/null || { echo "ERROR: rdkit is not installed"; exit 1; }
echo "✓ Dependencies OK"
echo ""

# Step 1: Extract archives
echo "Step 1: Extracting compressed safe data archives..."
echo ""

cd "$SCRIPT_DIR"
python convert_pickles.py unzip --base-dir .

echo ""

# Step 2: Convert to pickle
echo "Step 2: Converting safe format to pickle files..."
echo ""

python convert_pickles.py to_pickle --base-dir .

echo ""

# Step 3: Clean up safe format directories
echo "Step 3: Cleaning up safe format directories..."
echo ""

# Remove safe_* directories in runs/
if [ -d "$SCRIPT_DIR/runs" ]; then
    echo "Removing safe_* directories from runs/..."
    find "$SCRIPT_DIR/runs" -maxdepth 1 -type d -name "safe_*" -exec rm -rf {} +
fi

# Remove safe directories in comparison_algorithms/
if [ -d "$SCRIPT_DIR/comparison_algorithms/mcf_samples" ]; then
    echo "Removing safe directories from comparison_algorithms/mcf_samples/..."
    find "$SCRIPT_DIR/comparison_algorithms/mcf_samples" -maxdepth 1 -type d -name "ml-mcf_samples_random_split_*" -exec rm -rf {} +
fi

if [ -d "$SCRIPT_DIR/comparison_algorithms/geodiff_samples" ]; then
    echo "Removing safe directories from comparison_algorithms/geodiff_samples/..."
    find "$SCRIPT_DIR/comparison_algorithms/geodiff_samples" -maxdepth 1 -type d -name "geodiff_samples_random_split_*" -exec rm -rf {} +
fi

# Remove safe directories in runs/run_*/conformers/
echo "Removing safe directories from runs/run_*/conformers/..."
find "$SCRIPT_DIR/runs" -type d -path "*/conformers/safe_*" -exec rm -rf {} +

echo "✓ Cleanup complete"
echo ""
echo "=============================================================================="
echo "Setup complete!"
echo "=============================================================================="
