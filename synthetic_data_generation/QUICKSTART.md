# 🚀 Quick Start Guide - Synthetic Aerodynamic Dataset Generation

Get started generating F1 synthetic aerodynamic data in **5 minutes**.

---

## ⚡ Fastest Start (Windows)

```powershell
# 1. Navigate to directory
cd synthetic_data_generation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate 100 samples (test run)
python batch_orchestrator.py --tier1 100 --workers 4 --output ./test_dataset

# 4. Visualize results
python visualize_dataset.py --input ./test_dataset
```

**Done!** Check `./test_dataset/visualizations/` for plots.

---

## 📦 What You Get

After running, you'll have:

```
test_dataset/
├── scalars.json              # All aerodynamic coefficients (CL, CD, L/D, etc.)
├── field_data.h5             # Pressure distributions, circulation, etc.
├── summary.json              # Dataset statistics
├── generation_plan.json      # Sampling plan used
└── visualizations/
    ├── efficiency_map.png    # L/D vs CL scatter plot
    ├── drag_polar.png        # CL vs CD (with DRS states)
    ├── balance_distribution.png
    ├── parameter_sensitivity.png
    ├── correlation_matrix.png
    └── summary_report.txt
```

---

## 🎯 Common Use Cases

### 1. Quick Test (100 samples, ~8 minutes)

```bash
python batch_orchestrator.py --tier1 100 --workers 4
```

### 2. Small Training Set (1000 samples, ~1.5 hours)

```bash
python batch_orchestrator.py --tier1 1000 --workers 8
```

### 3. Production Dataset (5000 samples, ~7 hours)

```bash
python batch_orchestrator.py --tier1 5000 --workers 8 --output ./production_dataset
```

### 4. Using PowerShell Script (Windows)

```powershell
.\generate_dataset.ps1 -Tier1 1000 -Workers 8 -OutputDir ".\my_dataset"
```

### 5. Using Bash Script (Linux/Mac)

```bash
chmod +x generate_dataset.sh
./generate_dataset.sh 1000 0 0 8 ./my_dataset
```

---

## 🐳 Docker Quick Start

### Build and Run

```bash
# Build image
docker build -t f1-synthetic-data .

# Run generation (5000 samples)
docker run --rm \
  -v $(pwd)/data:/data \
  f1-synthetic-data \
  python3 batch_orchestrator.py --tier1 5000 --workers 8 --output /data/dataset
```

### Docker Compose (Multi-Tier)

```bash
# Generate Tier 1 data
docker-compose up tier1-generator

# View Dask dashboard (optional)
docker-compose up dask-scheduler dask-worker
# Open http://localhost:8787 in browser
```

---

## 📊 Viewing Results

### 1. Quick Statistics

```python
from batch_orchestrator import DatasetStorage

storage = DatasetStorage("./test_dataset")
stats = storage.get_statistics()

print(f"Samples: {stats['n_samples']}")
print(f"Mean CL: {stats['CL']['mean']:.3f}")
print(f"Mean L/D: {stats['L_over_D']['mean']:.2f}")
```

### 2. Load Single Sample

```python
import json

with open("./test_dataset/scalars.json") as f:
    samples = json.load(f)

# First sample
sample = samples[0]
print(f"CL: {sample['global_outputs']['CL']:.3f}")
print(f"CD: {sample['global_outputs']['CD_total']:.3f}")
print(f"Downforce: {sample['global_outputs']['L']:.0f} N")
```

### 3. Visualize in Paraview

```bash
# Generate VTK file from first sample
python -c "
from tier0_geometry import F1GeometryGenerator
from schema import GeometryParameters
import json

with open('./test_dataset/scalars.json') as f:
    samples = json.load(f)

geom = GeometryParameters(**samples[0]['geometry_params'])
generator = F1GeometryGenerator(geom)
mesh = generator.generate_complete_geometry()
mesh.save_vtk('sample_geometry.vtk')
print('Saved: sample_geometry.vtk')
"

# Open in Paraview
paraview sample_geometry.vtk
```

---

## 🔧 Customization

### Custom Parameter Ranges

Edit `schema.py` to change parameter bounds:

```python
@classmethod
def get_bounds(cls) -> Dict[str, tuple]:
    return {
        'main_plane_angle_deg': (-5.0, 10.0),  # Change these
        'rear_wing_angle_deg': (5.0, 25.0),
        # ... etc
    }
```

### Custom Sampling Strategy

```python
from sampling_strategy import ParameterSampler

sampler = ParameterSampler(seed=42)

# Generate custom samples
geom_samples = sampler.generate_geometry_samples(
    n_samples=500,
    method='lhs'  # or 'stratified'
)

# Use in your own loop
for geom in geom_samples:
    # Your custom processing
    pass
```

### Custom Flow Conditions

```python
from schema import FlowConditions

# High-speed scenario
flow = FlowConditions.standard_conditions(
    V_inf=90.0,  # m/s (very fast)
    ground_gap=15.0  # mm (low ride height)
)
flow.yaw = 3.0  # degrees (cornering)
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'numpy'"

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Generation is slow

**Solution:** Increase workers (use number of CPU cores):
```bash
python batch_orchestrator.py --tier1 1000 --workers 16
```

### Issue: Out of memory

**Solution:** Reduce workers or batch size:
```bash
python batch_orchestrator.py --tier1 1000 --workers 2
```

### Issue: VLM solver not converging

**Solution:** Check geometry parameters are within bounds. Extreme angles may cause issues.

### Issue: HDF5 file is huge

**Solution:** HDF5 uses compression. For even smaller files, reduce mesh resolution in `tier0_geometry.py`:
```python
# In generate_wing_section()
n_chord=10,  # Reduce from 15
n_span=15,   # Reduce from 25
```

---

## 📈 Performance Tips

### 1. Parallel Processing

Use all CPU cores:
```bash
# Check CPU count
python -c "import os; print(os.cpu_count())"

# Use all cores
python batch_orchestrator.py --workers 16
```

### 2. Batch Size

Generate in batches for large datasets:
```bash
# Generate 10k samples in 2 batches
python batch_orchestrator.py --tier1 5000 --output ./batch1
python batch_orchestrator.py --tier1 5000 --output ./batch2
```

### 3. Storage Optimization

Use Parquet instead of JSON for large datasets (TODO: implement in future version).

---

## 🎓 Next Steps

### 1. Train a Surrogate Model

```python
import torch
from torch.utils.data import Dataset, DataLoader
import json

class AeroDataset(Dataset):
    def __init__(self, dataset_dir):
        with open(f"{dataset_dir}/scalars.json") as f:
            self.samples = json.load(f)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Input: geometry parameters (16D)
        geom = list(sample['geometry_params'].values())
        X = torch.tensor(geom, dtype=torch.float32)
        
        # Output: CL, CD
        y = torch.tensor([
            sample['global_outputs']['CL'],
            sample['global_outputs']['CD_total']
        ], dtype=torch.float32)
        
        return X, y

# Create DataLoader
dataset = AeroDataset("./test_dataset")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train your model
for X_batch, y_batch in loader:
    # Your training code here
    pass
```

### 2. Integrate with Quantum Optimizer

Use generated data to train quantum-enhanced optimization:

```python
from quantum_service.vqe.optimizer import VQEOptimizer

# Load best configurations from dataset
# Use as initial population for quantum optimization
```

### 3. Real-Time Simulation

Integrate with real-time server:

```python
from realtime_server import AeroSimulationServer

# Use VLM solver for real-time predictions
# Serve via WebSocket to frontend
```

---

## 📚 Learn More

- **Full Documentation**: See [README.md](README.md)
- **Data Schema**: See [schema.py](schema.py)
- **VLM Theory**: Check `tier1_vlm_solver.py` comments
- **Sampling Methods**: See [sampling_strategy.py](sampling_strategy.py)

---

## 🤝 Need Help?

- **Check logs**: Look for error messages in console output
- **Validate data**: Run `python visualize_dataset.py --input <dir>`
- **Test small first**: Always test with `--tier1 10` before large runs
- **Check disk space**: ~2GB per 5000 samples

---

## ✅ Validation Checklist

Before using dataset for ML training:

- [ ] Generated at least 1000 samples
- [ ] Visualizations look reasonable (no outliers)
- [ ] CL range: 2.5 - 4.5 ✓
- [ ] CD range: 0.7 - 1.2 ✓
- [ ] L/D range: 3.0 - 6.0 ✓
- [ ] Balance: 35-45% front ✓
- [ ] No NaN or Inf values
- [ ] HDF5 file readable

---

**Ready to generate?** Run this now:

```bash
python batch_orchestrator.py --tier1 100 --workers 4
```

**Questions?** Check the [README.md](README.md) or open an issue.
