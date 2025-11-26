# 🚀 Quantum-Aero F1 Deployment Guide

**Complete deployment guide for production F1 integration**

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Service Configuration](#service-configuration)
4. [Starting Services](#starting-services)
5. [Testing & Validation](#testing--validation)
6. [Production Deployment](#production-deployment)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## 🖥️ System Requirements

### **Hardware Requirements**

**Development/Testing**:
- CPU: 8+ cores (Intel/AMD)
- RAM: 32GB minimum
- GPU: NVIDIA RTX 3080 or better (12GB+ VRAM)
- Storage: 500GB SSD

**Production**:
- CPU: 16+ cores
- RAM: 64GB minimum
- GPU: NVIDIA A100 or better (40GB+ VRAM)
- Storage: 2TB NVMe SSD
- Network: 10Gbps

### **Software Requirements**

- **OS**: Ubuntu 20.04+ or macOS 12+
- **Python**: 3.9+
- **Node.js**: 16+
- **Docker**: 20.10+
- **CUDA**: 11.8+ (for GPU)

### **External Services**

- **IBM Quantum**: Account + API token
- **D-Wave Leap**: Account + API token
- **NVIDIA Omniverse**: License (optional)
- **Apache Kafka**: 3.0+ (production)
- **TimescaleDB**: 2.0+ (production)

---

## 📦 Installation

### **Step 1: Clone Repository**

```bash
git clone https://github.com/rjamoriz/Quantum-Aero-F1-Prototype.git
cd Quantum-Aero-F1-Prototype
```

### **Step 2: Run Setup Script**

```bash
chmod +x setup_evolution.sh
./setup_evolution.sh
```

This installs:
- Python dependencies (PyTorch, Qiskit, D-Wave Ocean, etc.)
- Node.js dependencies (React, Recharts, etc.)
- System dependencies

### **Step 3: Configure Environment**

```bash
cp agents/.env.template agents/.env
vim agents/.env
```

Add your API tokens:
```bash
IBM_QUANTUM_TOKEN=your-ibm-token-here
DWAVE_API_TOKEN=your-dwave-token-here
OPENAI_API_KEY=your-openai-key-here  # Optional
```

### **Step 4: Generate Data**

```bash
# Generate synthetic CFD dataset
python data_generation/synthetic_cfd_generator.py \
  --num-train 10000 \
  --num-val 2000 \
  --num-test 1000
```

---

## ⚙️ Service Configuration

### **Service Ports**

| Service | Port | Description |
|---------|------|-------------|
| API Gateway | 8000 | Unified API access |
| AeroTransformer | 8003 | Vision Transformer CFD |
| GNN-RANS | 8004 | Graph neural networks |
| VQE Quantum | 8005 | IBM Quantum VQE |
| D-Wave | 8006 | Quantum annealing |
| Diffusion Models | 8007 | 3D geometry generation |
| RL Control | 8008 | PPO optimization |
| AeroGAN | 8009 | StyleGAN3 design |
| Frontend | 3000 | React dashboard |

### **Configuration Files**

- `agents/.env` - API tokens and secrets
- `config/services.yaml` - Service configuration
- `config/quantum.yaml` - Quantum hardware settings
- `config/production.yaml` - Production settings

---

## 🚀 Starting Services

### **Option 1: All Services (Development)**

```bash
# Terminal 1: API Gateway
python api_gateway.py

# Terminal 2: ML Services
python -m ml_service.models.aero_transformer.api &
python -m ml_service.models.gnn_rans.api &

# Terminal 3: Quantum Services
python -m quantum_service.vqe.api &
python -m quantum_service.dwave.api &

# Terminal 4: Generative Services
python -m ml_service.models.diffusion.api &
python -m ml_service.rl.api &
python -m ml_service.models.aerogan.api &

# Terminal 5: Frontend
cd frontend
npm start
```

### **Option 2: Docker Compose (Production)**

```bash
docker-compose up -d
```

### **Option 3: Kubernetes (Scale)**

```bash
kubectl apply -f k8s/
```

---

## ✅ Testing & Validation

### **Run Test Suite**

```bash
# Unit tests
pytest tests/ -v

# Integration tests
pytest tests/test_integration.py -v

# Complete system tests
pytest tests/test_complete_system.py -v

# Performance benchmarks
python tests/benchmark_all.py
```

### **Validate Services**

```bash
# Check all services are running
curl http://localhost:8000/health

# Test AeroTransformer
curl -X POST http://localhost:8003/api/ml/aerotransformer/predict \
  -H "Content-Type: application/json" \
  -d '{"geometry": [[1,2,3]], "conditions": {"velocity": 50}}'

# Test VQE Quantum
curl -X POST http://localhost:8005/api/quantum/vqe/optimize \
  -H "Content-Type: application/json" \
  -d '{"num_variables": 20, "target_cl": 2.8}'

# Test D-Wave
curl -X POST http://localhost:8006/api/quantum/dwave/optimize-wing \
  -H "Content-Type: application/json" \
  -d '{"num_elements": 50, "target_cl": 2.8}'
```

### **Performance Validation**

```bash
# Run performance tests
python tests/validate_performance.py

# Expected results:
# ✓ AeroTransformer: <50ms
# ✓ GNN-RANS: 1000x speedup
# ✓ VQE: <100 circuit depth
# ✓ D-Wave: 5000+ variables
# ✓ Diffusion: <5s generation
# ✓ Digital Twin: <100ms sync
# ✓ Telemetry: <1s optimization
```

---

## 🏭 Production Deployment

### **Pre-Deployment Checklist**

- [ ] All tests passing
- [ ] Performance targets met
- [ ] API tokens configured
- [ ] Database migrations complete
- [ ] SSL certificates installed
- [ ] Monitoring configured
- [ ] Backup strategy in place
- [ ] Load balancer configured

### **Deployment Steps**

#### **1. Database Setup**

```bash
# TimescaleDB for telemetry
docker run -d --name timescaledb \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=secure_password \
  timescale/timescaledb:latest-pg14

# Initialize schema
psql -h localhost -U postgres -d f1_telemetry -f sql/init.sql
```

#### **2. Kafka Setup**

```bash
# Start Kafka cluster
docker-compose -f kafka/docker-compose.yml up -d

# Create topics
kafka-topics --create --topic f1_telemetry \
  --bootstrap-server localhost:9092 \
  --partitions 10 --replication-factor 3
```

#### **3. Deploy Services**

```bash
# Build Docker images
docker build -t f1-api-gateway:latest .
docker build -t f1-ml-services:latest -f Dockerfile.ml .
docker build -t f1-quantum:latest -f Dockerfile.quantum .

# Deploy to Kubernetes
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/services/
kubectl apply -f k8s/deployments/
kubectl apply -f k8s/ingress.yaml
```

#### **4. Configure Load Balancer**

```bash
# NGINX configuration
upstream api_gateway {
    server 10.0.1.10:8000;
    server 10.0.1.11:8000;
    server 10.0.1.12:8000;
}

server {
    listen 443 ssl;
    server_name api.f1-quantum-aero.com;
    
    ssl_certificate /etc/ssl/certs/f1-quantum.crt;
    ssl_certificate_key /etc/ssl/private/f1-quantum.key;
    
    location / {
        proxy_pass http://api_gateway;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### **5. Enable Monitoring**

```bash
# Prometheus + Grafana
kubectl apply -f monitoring/prometheus.yaml
kubectl apply -f monitoring/grafana.yaml

# Access Grafana
kubectl port-forward svc/grafana 3000:3000
# Login: admin/admin
```

---

## 📊 Monitoring

### **Key Metrics**

**Service Health**:
- Response time (p50, p95, p99)
- Error rate
- Request throughput
- CPU/Memory usage

**ML Models**:
- Inference latency
- Prediction accuracy
- Model drift
- GPU utilization

**Quantum Services**:
- Queue position
- Job success rate
- Circuit depth
- Quantum volume

**Production Systems**:
- Digital twin sync latency
- Telemetry throughput
- Optimization latency
- Data pipeline health

### **Grafana Dashboards**

```bash
# Import dashboards
curl -X POST http://localhost:3000/api/dashboards/import \
  -H "Content-Type: application/json" \
  -d @monitoring/dashboards/f1-overview.json
```

### **Alerts**

```yaml
# Prometheus alerts
groups:
  - name: f1_alerts
    rules:
      - alert: HighLatency
        expr: http_request_duration_seconds{quantile="0.95"} > 0.1
        for: 5m
        annotations:
          summary: "High API latency detected"
      
      - alert: QuantumJobFailed
        expr: quantum_job_failures_total > 5
        for: 10m
        annotations:
          summary: "Multiple quantum job failures"
```

---

## 🐛 Troubleshooting

### **Common Issues**

#### **Service Won't Start**

```bash
# Check logs
docker logs <container-id>

# Check port conflicts
lsof -i :8000

# Verify dependencies
pip list | grep torch
```

#### **Quantum Connection Failed**

```bash
# Test IBM Quantum
python << EOF
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(token="your-token")
print(service.backends())
EOF

# Test D-Wave
dwave ping --client qpu
```

#### **High Latency**

```bash
# Check GPU utilization
nvidia-smi

# Profile service
python -m cProfile -o profile.stats api_gateway.py

# Analyze
python -m pstats profile.stats
```

#### **Memory Issues**

```bash
# Monitor memory
watch -n 1 free -h

# Check for leaks
python -m memory_profiler ml_service/models/aero_transformer/api.py

# Increase limits
ulimit -v unlimited
```

---

## 📞 Support

- **Documentation**: https://docs.f1-quantum-aero.com
- **Issues**: https://github.com/rjamoriz/Quantum-Aero-F1-Prototype/issues
- **Email**: support@f1-quantum-aero.com

---

## 🎯 Quick Reference

### **Start Everything**

```bash
./start_all.sh
```

### **Stop Everything**

```bash
./stop_all.sh
```

### **View Logs**

```bash
./logs.sh [service-name]
```

### **Run Tests**

```bash
./test.sh
```

### **Deploy to Production**

```bash
./deploy.sh production
```

---

**🚀 Ready for production F1 deployment!**
