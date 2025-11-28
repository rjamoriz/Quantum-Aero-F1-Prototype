# PowerShell script to generate synthetic aerodynamic dataset
# Usage: .\generate_dataset.ps1 -Tier1 1000 -Workers 4

param(
    [int]$Tier1 = 1000,
    [int]$Tier2 = 0,
    [int]$Tier3 = 0,
    [int]$Workers = 4,
    [string]$OutputDir = ".\synthetic_dataset"
)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "F1 Synthetic Dataset Generation" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Tier 1 samples: $Tier1"
Write-Host "  Tier 2 samples: $Tier2"
Write-Host "  Tier 3 samples: $Tier3"
Write-Host "  Workers: $Workers"
Write-Host "  Output: $OutputDir"
Write-Host ""

# Check Python installation
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Error: Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check dependencies
Write-Host "`nChecking dependencies..." -ForegroundColor Yellow
$checkDeps = python -c "import numpy, scipy, h5py, tqdm" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Generate sampling plan
Write-Host "`nGenerating sampling plan..." -ForegroundColor Yellow
python -c @"
from sampling_strategy import DatasetPlan
planner = DatasetPlan()
planner.save_plan('$OutputDir/generation_plan.json')
"@

# Run generation
Write-Host "`nStarting dataset generation..." -ForegroundColor Yellow
python batch_orchestrator.py `
    --tier1 $Tier1 `
    --tier2 $Tier2 `
    --tier3 $Tier3 `
    --workers $Workers `
    --output "$OutputDir"

if ($LASTEXITCODE -eq 0) {
    # Generate summary report
    Write-Host "`nGenerating summary report..." -ForegroundColor Yellow
    python -c @"
from batch_orchestrator import DatasetStorage
import json

storage = DatasetStorage('$OutputDir')
stats = storage.get_statistics()

print('\n' + '='*60)
print('DATASET GENERATION COMPLETE')
print('='*60)
print(f'\nTotal samples: {stats["n_samples"]}')
print(f'Output directory: $OutputDir')
print('\nStatistics:')
print(f'  CL: {stats["CL"]["mean"]:.3f} ± {stats["CL"]["std"]:.3f}')
print(f'  CD: {stats["CD"]["mean"]:.3f} ± {stats["CD"]["std"]:.3f}')
print(f'  L/D: {stats["L_over_D"]["mean"]:.2f} ± {stats["L_over_D"]["std"]:.2f}')
print('='*60)

# Save summary
with open('$OutputDir/summary.json', 'w') as f:
    json.dump(stats, f, indent=2)
"@

    Write-Host ""
    Write-Host "✓ Dataset saved to: $OutputDir" -ForegroundColor Green
    Write-Host "✓ Summary: $OutputDir\summary.json" -ForegroundColor Green
    Write-Host "✓ Scalars: $OutputDir\scalars.json" -ForegroundColor Green
    Write-Host "✓ Fields: $OutputDir\field_data.h5" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Visualize: python visualize_dataset.py --input $OutputDir"
    Write-Host "  2. Train ML: Use scalars.json and field_data.h5 for training"
    Write-Host "  3. Validate: Check summary.json for quality metrics"
} else {
    Write-Host "`n✗ Dataset generation failed" -ForegroundColor Red
    exit 1
}
