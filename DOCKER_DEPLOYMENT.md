# 🚀 Docker Deployment Guide

## Quick Start

### Prerequisites
- Docker 24.0+ installed
- Docker Compose 2.20+ installed
- 8GB RAM minimum (16GB recommended)
- Ports available: 3000, 3001, 8001, 27017, 6379, 4222, 8222

### One-Command Start

```bash
./start_platform.sh
```

### Manual Start

```bash
# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

## 🌐 Services & Ports

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| **Frontend** | 3000 | http://localhost:3000 | React UI with VLM visualization |
| **Backend API** | 3001 | http://localhost:3001 | REST API for data management |
| **Physics Engine** | 8001 | http://localhost:8001 | VLM Solver & Aerodynamics |
| **NATS Monitoring** | 8222 | http://localhost:8222 | Message broker dashboard |
| **MongoDB** | 27017 | mongodb://localhost:27017 | Database |
| **Redis** | 6379 | redis://localhost:6379 | Cache & queue |
| **NATS** | 4222 | nats://localhost:4222 | Message broker |

## 🧪 Testing the Platform

### 1. Check Services Health

```bash
# Frontend
curl http://localhost:3000/health

# Backend
curl http://localhost:3001/health

# Physics Engine (VLM)
curl http://localhost:8001/health

# NATS
curl http://localhost:8222/healthz
```

### 2. Test VLM Solver

```bash
curl -X POST http://localhost:8001/vlm/solve \
  -H "Content-Type: application/json" \
  -d '{
    "geometry": {
      "span": 1.8,
      "chord": 0.25,
      "twist": -2.0,
      "dihedral": 0.0,
      "sweep": 0.0,
      "taper_ratio": 1.0
    },
    "velocity": 50,
    "alpha": 5.0,
    "yaw": 0.0,
    "rho": 1.225,
    "n_panels_x": 20,
    "n_panels_y": 10
  }'
```

Expected response:
```json
{
  "cl": 0.55,
  "cd": 0.025,
  "cm": -0.15,
  "l_over_d": 22.0,
  "lift": 850.5,
  "drag": 38.7,
  ...
}
```

### 3. Access Frontend

Open browser: http://localhost:3000

Features available:
- ✅ NACA Airfoil selection (6412, 4415, 4418, etc.)
- ✅ VLM visualization with 3D lattice
- ✅ Aerodynamic data generation (Lift, Drag, Pressure)
- ✅ Reynolds number configuration
- ✅ Quantum optimization dashboard (QUBO)
- ✅ Multi-physics analysis (Thermal, Acoustic, Aeroelastic)

## 🔧 Advanced Configuration

### Start with GenAI Agents

```bash
# Start core platform first
docker-compose up -d

# Then start AI agents
export ANTHROPIC_API_KEY="your-api-key-here"
docker-compose -f docker-compose.agents.yml up -d
```

Agents include:
- Master Orchestrator (Claude Sonnet 4.5)
- ML Surrogate Agent
- Quantum Optimizer Agent
- Physics Validator Agent
- Analysis Agent

### Enable Monitoring

```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

Access:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (admin/admin)

## 📊 Component Overview

### Frontend (React + Three.js)
- **Components**: 
  - VLMVisualization.jsx - 3D vortex lattice visualization
  - AerodynamicDataGenerator.jsx - NACA airfoil data generation
  - QuantumOptimizationDashboard.jsx - QUBO optimization
  - ThermalAnalysisPanel.jsx - Thermal analysis
  - AeroacousticAnalysisPanel.jsx - Acoustic analysis
  - AeroelasticAnalysisPanel.jsx - Flutter analysis

### Backend Services
- **Physics Engine**: VLM solver (Python/FastAPI)
- **Backend API**: Data management (Node.js/Express)
- **Message Broker**: NATS for agent communication

### Data Flow
```
User (Browser)
    ↓
Frontend (React) :3000
    ↓
Backend API :3001
    ↓
Physics Engine (VLM) :8001
    ↓
Results → MongoDB
    ↓
Quantum Optimization (QUBO)
```

## 🐛 Troubleshooting

### Services not starting
```bash
# Check Docker status
docker ps

# View logs
docker-compose logs -f physics-engine
docker-compose logs -f backend
docker-compose logs -f frontend

# Restart specific service
docker-compose restart physics-engine
```

### Port conflicts
```bash
# Check what's using a port
lsof -i :3000
lsof -i :8001

# Kill process
kill -9 <PID>
```

### Frontend not connecting to backend
```bash
# Check nginx proxy configuration
docker exec qaero-frontend cat /etc/nginx/conf.d/default.conf

# Restart frontend
docker-compose restart frontend
```

### VLM solver errors
```bash
# Check physics-engine logs
docker-compose logs -f physics-engine

# Test solver directly
curl http://localhost:8001/vlm/validate
```

## 🔄 Development Workflow

### Hot Reload Frontend
```bash
# Stop containerized frontend
docker-compose stop frontend

# Run frontend locally with hot reload
cd frontend
npm install
npm start
# Access at http://localhost:3000
```

### Rebuild Specific Service
```bash
# Rebuild and restart physics engine
docker-compose up -d --build physics-engine

# View build logs
docker-compose logs -f physics-engine
```

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale ML agents
docker-compose -f docker-compose.agents.yml up -d --scale ml-agent=5

# Scale physics engines
docker-compose up -d --scale physics-engine=3
```

## 🛑 Shutdown

```bash
# Graceful shutdown
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## 📝 Environment Variables

Create `.env` file:
```env
# API Keys
ANTHROPIC_API_KEY=sk-ant-xxxxx

# Database
MONGODB_URI=mongodb://mongodb:27017/qaero
REDIS_URL=redis://redis:6379

# Services
PHYSICS_SERVICE_URL=http://physics-engine:8001
BACKEND_URL=http://backend:3001

# Monitoring
GRAFANA_PASSWORD=secure_password
```

## ✅ What Works Out of the Box

✅ **NACA Airfoils**: 6412, 4415, 4418, 9618, 0009, 23012
✅ **VLM Simulation**: Full vortex lattice method with circulation
✅ **Aerodynamic Data**: Lift (CL), Drag (CD), Pressure distribution
✅ **Reynolds Number**: Configurable viscosity effects
✅ **3D Visualization**: Interactive lattice, wake vortices, velocity vectors
✅ **Quantum Optimization**: QUBO for drag minimization & downforce maximization
✅ **Multi-Physics**: Thermal, acoustic, aeroelastic analysis
✅ **Export**: JSON data export for all analyses

## 🎯 Next Steps

After platform starts:
1. Open http://localhost:3000
2. Select "Aerodinámica" tab
3. Choose NACA profile and F1 component
4. Generate VLM data
5. Visualize 3D lattice and results
6. Open "Quantum" tab for QUBO optimization
7. Minimize drag / Maximize downforce

**Everything is integrated and ready to use!** 🚀
