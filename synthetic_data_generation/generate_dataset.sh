#!/bin/bash
# Shell script to generate synthetic aerodynamic dataset
# Usage: ./generate_dataset.sh [tier1_samples] [tier2_samples] [tier3_samples]

set -e  # Exit on error

# Default values
TIER1_SAMPLES=${1:-1000}
TIER2_SAMPLES=${2:-0}
TIER3_SAMPLES=${3:-0}
WORKERS=${4:-4}
OUTPUT_DIR=${5:-"./synthetic_dataset"}

echo "=========================================="
echo "F1 Synthetic Dataset Generation"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Tier 1 samples: $TIER1_SAMPLES"
echo "  Tier 2 samples: $TIER2_SAMPLES"
echo "  Tier 3 samples: $TIER3_SAMPLES"
echo "  Workers: $WORKERS"
echo "  Output: $OUTPUT_DIR"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
python3 -c "import numpy, scipy, h5py, tqdm" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -r requirements.txt
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate sampling plan
echo ""
echo "Generating sampling plan..."
python3 -c "
from sampling_strategy import DatasetPlan
planner = DatasetPlan()
planner.save_plan('$OUTPUT_DIR/generation_plan.json')
"

# Run generation
echo ""
echo "Starting dataset generation..."
python3 batch_orchestrator.py \
    --tier1 $TIER1_SAMPLES \
    --tier2 $TIER2_SAMPLES \
    --tier3 $TIER3_SAMPLES \
    --workers $WORKERS \
    --output "$OUTPUT_DIR"

# Generate summary report
echo ""
echo "Generating summary report..."
python3 -c "
from batch_orchestrator import DatasetStorage
import json

storage = DatasetStorage('$OUTPUT_DIR')
stats = storage.get_statistics()

print('\n' + '='*60)
print('DATASET GENERATION COMPLETE')
print('='*60)
print(f'\nTotal samples: {stats[\"n_samples\"]}')
print(f'Output directory: $OUTPUT_DIR')
print('\nStatistics:')
print(f'  CL: {stats[\"CL\"][\"mean\"]:.3f} ± {stats[\"CL\"][\"std\"]:.3f}')
print(f'  CD: {stats[\"CD\"][\"mean\"]:.3f} ± {stats[\"CD\"][\"std\"]:.3f}')
print(f'  L/D: {stats[\"L_over_D\"][\"mean\"]:.2f} ± {stats[\"L_over_D\"][\"std\"]:.2f}')
print('='*60)

# Save summary
with open('$OUTPUT_DIR/summary.json', 'w') as f:
    json.dump(stats, f, indent=2)
"

echo ""
echo "✓ Dataset saved to: $OUTPUT_DIR"
echo "✓ Summary: $OUTPUT_DIR/summary.json"
echo "✓ Scalars: $OUTPUT_DIR/scalars.json"
echo "✓ Fields: $OUTPUT_DIR/field_data.h5"
echo ""
echo "Next steps:"
echo "  1. Visualize: python3 visualize_dataset.py --input $OUTPUT_DIR"
echo "  2. Train ML: Use scalars.json and field_data.h5 for training"
echo "  3. Validate: Check summary.json for quality metrics"
