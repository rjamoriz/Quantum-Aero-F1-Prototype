# 🚀 Quantum Hardware Deployment Guide

**Production deployment to IBM Quantum and D-Wave systems**

---

## 📋 Overview

This guide covers deploying the Quantum-Aero F1 platform to real quantum hardware:
- **IBM Quantum System One** (VQE optimization)
- **D-Wave Advantage** (Quantum annealing)

---

## 🔐 Prerequisites

### **1. IBM Quantum Account**

```bash
# Sign up at: https://quantum-computing.ibm.com/

# Get your API token from:
# Account → API Token

# Save token
export IBM_QUANTUM_TOKEN="your-token-here"
```

### **2. D-Wave Leap Account**

```bash
# Sign up at: https://cloud.dwavesys.com/leap/

# Get your API token from:
# Account → API Token

# Save token
export DWAVE_API_TOKEN="your-token-here"
```

### **3. Install Quantum SDKs**

```bash
# IBM Qiskit
pip install qiskit>=1.0.0
pip install qiskit-ibm-runtime>=0.17.0
pip install qiskit-aer>=0.13.0

# D-Wave Ocean SDK
pip install dwave-ocean-sdk>=6.7.0
pip install dwave-system>=1.22.0
pip install dwave-samplers>=1.2.0
```

---

## 🔧 Configuration

### **1. Update Environment Variables**

Edit `agents/.env`:

```bash
# IBM Quantum
IBM_QUANTUM_TOKEN=your-ibm-token-here
IBM_QUANTUM_BACKEND=ibm_brisbane  # or ibm_kyoto, ibm_osaka
IBM_QUANTUM_HUB=ibm-q
IBM_QUANTUM_GROUP=open
IBM_QUANTUM_PROJECT=main

# D-Wave
DWAVE_API_TOKEN=your-dwave-token-here
DWAVE_SOLVER=Advantage_system6.1  # Latest Pegasus topology
DWAVE_ENDPOINT=https://cloud.dwavesys.com/sapi
```

### **2. Update VQE Service**

Edit `quantum_service/vqe/optimizer.py`:

```python
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session

class VQEAeroOptimizer:
    def __init__(self, use_hardware: bool = True):
        if use_hardware:
            # Connect to IBM Quantum
            service = QiskitRuntimeService(
                channel="ibm_quantum",
                token=os.getenv('IBM_QUANTUM_TOKEN')
            )
            
            # Get backend
            backend = service.backend(os.getenv('IBM_QUANTUM_BACKEND', 'ibm_brisbane'))
            
            # Create session
            self.session = Session(backend=backend)
            self.estimator = Estimator(session=self.session)
```

### **3. Update D-Wave Service**

Edit `quantum_service/dwave/annealer.py`:

```python
from dwave.system import DWaveSampler, EmbeddingComposite
import os

class DWaveAeroAnnealer:
    def __init__(self, use_hardware: bool = True):
        if use_hardware:
            # Connect to D-Wave
            self.sampler = EmbeddingComposite(
                DWaveSampler(
                    token=os.getenv('DWAVE_API_TOKEN'),
                    solver=os.getenv('DWAVE_SOLVER', 'Advantage_system6.1'),
                    endpoint=os.getenv('DWAVE_ENDPOINT')
                )
            )
```

---

## 🧪 Testing Hardware Connection

### **Test IBM Quantum**

```python
# test_ibm_connection.py
from qiskit_ibm_runtime import QiskitRuntimeService
import os

# Load token
service = QiskitRuntimeService(
    channel="ibm_quantum",
    token=os.getenv('IBM_QUANTUM_TOKEN')
)

# List available backends
print("Available IBM Quantum backends:")
for backend in service.backends():
    print(f"  - {backend.name}: {backend.num_qubits} qubits")

# Get specific backend
backend = service.backend('ibm_brisbane')
print(f"\nSelected: {backend.name}")
print(f"  Qubits: {backend.num_qubits}")
print(f"  Status: {backend.status().status_msg}")
```

### **Test D-Wave Connection**

```python
# test_dwave_connection.py
from dwave.system import DWaveSampler
import os

# Connect
sampler = DWaveSampler(
    token=os.getenv('DWAVE_API_TOKEN'),
    solver='Advantage_system6.1'
)

# Get properties
print("D-Wave Advantage Properties:")
print(f"  Solver: {sampler.properties['chip_id']}")
print(f"  Qubits: {sampler.properties['num_qubits']}")
print(f"  Topology: {sampler.properties['topology']['type']}")
print(f"  Status: {sampler.properties['status']}")
```

---

## 🚀 Deployment Steps

### **Step 1: Generate Training Data**

```bash
# Generate synthetic CFD dataset
python data_generation/synthetic_cfd_generator.py

# This creates:
# - data/cfd_dataset/train/ (10,000 samples)
# - data/cfd_dataset/val/ (2,000 samples)
# - data/cfd_dataset/test/ (1,000 samples)
```

### **Step 2: Train ML Models**

```bash
# Train AeroTransformer
python -m ml_service.models.aero_transformer.train

# Train GNN-RANS (requires PyTorch Geometric)
python -m ml_service.models.gnn_rans.train
```

### **Step 3: Start Services with Hardware**

```bash
# Export tokens
export IBM_QUANTUM_TOKEN="your-ibm-token"
export DWAVE_API_TOKEN="your-dwave-token"

# Start VQE service (IBM Quantum)
python -m quantum_service.vqe.api --hardware

# Start D-Wave service
python -m quantum_service.dwave.api --hardware

# Start other services
python -m ml_service.models.aero_transformer.api
python -m ml_service.models.gnn_rans.api

# Start API gateway
python api_gateway.py
```

### **Step 4: Run Hardware Tests**

```bash
# Test VQE on IBM Quantum
curl -X POST http://localhost:8005/api/quantum/vqe/optimize-aero \
  -H "Content-Type: application/json" \
  -d '{
    "num_variables": 20,
    "target_cl": 2.8,
    "target_cd": 0.4
  }'

# Test D-Wave annealing
curl -X POST http://localhost:8006/api/quantum/dwave/optimize-wing \
  -H "Content-Type: application/json" \
  -d '{
    "num_elements": 50,
    "target_cl": 2.8,
    "target_cd": 0.4,
    "num_reads": 1000
  }'
```

---

## 📊 Data Pipeline

### **1. CFD Data Generation**

```python
from data_generation.synthetic_cfd_generator import SyntheticCFDGenerator

# Create generator
generator = SyntheticCFDGenerator(
    volume_size=(64, 64, 64),
    output_dir='data/cfd_dataset'
)

# Generate dataset
generator.generate_dataset(
    num_train=10000,
    num_val=2000,
    num_test=1000
)
```

### **2. Quantum Data Encoding**

```python
from quantum_service.data_encoding import QuantumDataEncoder

# Create encoder
encoder = QuantumDataEncoder(num_bits_per_variable=4)

# Encode aerodynamic parameters
parameters = {
    'angle_of_attack': 5.0,
    'camber': 0.04,
    'thickness': 0.12
}

bounds = {
    'angle_of_attack': (-10.0, 20.0),
    'camber': (0.0, 0.08),
    'thickness': (0.08, 0.16)
}

# Encode to binary
encoded = encoder.encode_aerodynamic_parameters(parameters, bounds)

# Create QUBO for D-Wave
qubo = encoder.aerodynamic_to_qubo(num_design_vars=20)

# Create Ising Hamiltonian for VQE
h, J, offset = encoder.create_ising_hamiltonian(qubo)
```

### **3. Result Decoding**

```python
from quantum_service.data_encoding import QuantumResultDecoder

# Create decoder
decoder = QuantumResultDecoder()

# Decode D-Wave solution
parameters = decoder.decode_dwave_solution(
    solution={0: 1, 1: 0, 2: 1, ...},
    parameter_mapping={'angle_of_attack': [0, 1, 2, 3], ...},
    bounds=bounds
)

# Compute aerodynamic coefficients
coefficients = decoder.compute_aerodynamic_coefficients(parameters)
print(f"Cl: {coefficients['cl']:.3f}")
print(f"Cd: {coefficients['cd']:.3f}")
```

---

## 💰 Cost Estimation

### **IBM Quantum**
- **Free tier**: 10 minutes/month QPU time
- **Premium**: $1.60/second QPU time
- **Typical VQE run**: 1-5 seconds
- **Cost per optimization**: $1.60 - $8.00

### **D-Wave Leap**
- **Free tier**: 1 minute/month QPU time
- **Developer**: $2,000/month (unlimited)
- **Typical annealing**: 20 μs per read
- **Cost per 1000 reads**: ~$0.02

### **Monthly Budget**
- **Development**: $0 (simulators)
- **Testing**: ~$100/month (free tiers)
- **Production**: ~$2,500/month (IBM + D-Wave)

---

## 🔍 Monitoring

### **IBM Quantum Jobs**

```python
# Check job status
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
job = service.job('job-id-here')

print(f"Status: {job.status()}")
print(f"Queue position: {job.queue_position()}")
print(f"Estimated start: {job.queue_info().estimated_start_time}")
```

### **D-Wave Jobs**

```python
# Check solver status
from dwave.system import DWaveSampler

sampler = DWaveSampler()
print(f"Status: {sampler.properties['status']}")
print(f"Queue length: {sampler.properties['queue_length']}")
```

---

## 🐛 Troubleshooting

### **IBM Quantum Issues**

**Problem**: `IBMAccountError: No active account`
```bash
# Solution: Save credentials
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(token='your-token', overwrite=True)
```

**Problem**: `Backend not available`
```bash
# Solution: Check backend status
service = QiskitRuntimeService()
backend = service.backend('ibm_brisbane')
print(backend.status())
```

### **D-Wave Issues**

**Problem**: `SolverAuthenticationError`
```bash
# Solution: Configure token
dwave config create
# Enter token when prompted
```

**Problem**: `ChainBreakError`
```bash
# Solution: Increase chain strength
sampler.sample(bqm, chain_strength=2.0)
```

---

## 📈 Performance Optimization

### **VQE Optimization**
- Use warm-start from ML predictions
- Reduce circuit depth (fewer layers)
- Use error mitigation techniques
- Batch multiple problems

### **D-Wave Optimization**
- Optimize embedding (use `minorminer`)
- Tune chain strength
- Increase num_reads for better statistics
- Use hybrid solvers for large problems

---

## 🎯 Success Metrics

### **VQE (IBM Quantum)**
- ✅ Circuit depth < 100
- ✅ Convergence in < 200 iterations
- ✅ Solution quality > 95%
- ✅ QPU time < 5 seconds

### **D-Wave Annealing**
- ✅ Chain break fraction < 5%
- ✅ Problem size > 1000 variables
- ✅ Solution quality > 90%
- ✅ Annealing time < 100 μs

---

## 📚 Resources

- **IBM Quantum**: https://quantum-computing.ibm.com/
- **D-Wave Leap**: https://cloud.dwavesys.com/leap/
- **Qiskit Docs**: https://qiskit.org/documentation/
- **Ocean SDK Docs**: https://docs.ocean.dwavesys.com/

---

**🚀 Ready for quantum hardware deployment!**
