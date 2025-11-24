# Quantum-Aero F1 Prototype Setup Guide

**Complete guide to deploy the GenAI-powered multi-agent aerodynamic optimization platform**

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Agent Deployment](#agent-deployment)
5. [Frontend Integration](#frontend-integration)
6. [Verification & Testing](#verification--testing)
7. [Monitoring Setup](#monitoring-setup)
8. [Troubleshooting](#troubleshooting)

---

## 🖥️ System Requirements

### Hardware

**Minimum (Development):**
- CPU: 8 cores
- RAM: 16 GB
- GPU: NVIDIA RTX 3060 (6GB VRAM) or better
- Storage: 100 GB SSD

**Recommended (Production):**
- CPU: 16+ cores
- RAM: 64 GB
- GPU: NVIDIA RTX 4070/4090 (12GB+ VRAM)
- Storage: 500 GB NVMe SSD

### Software

- **OS:** Ubuntu 22.04 LTS or later
- **Docker:** 24.0+ with Docker Compose v2
- **NVIDIA Drivers:** 535+ with CUDA 12.3+
- **Python:** 3.11+
- **Node.js:** 18+
- **Git:** Latest

---

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Clone repository
git clone https://github.com/your-org/Quantum-Aero-F1-Prototype.git
cd Quantum-Aero-F1-Prototype

# 2. Set up environment
cp agents/.env.example agents/.env
# Edit agents/.env and add your ANTHROPIC_API_KEY

# 3. Start core infrastructure
docker-compose -f docker-compose.agents.yml up -d nats mongodb redis qdrant

# 4. Start agents
docker-compose -f docker-compose.agents.yml up -d master-orchestrator ml-agent

# 5. Verify
curl http://localhost:6001/health
# Expected: {"status": "healthy"}
```

---

## 📦 Detailed Setup

### Step 1: Install Prerequisites

#### Ubuntu 22.04

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3-pip

# Install Node.js 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### Step 2: Get Anthropic API Key

1. Sign up at https://console.anthropic.com/
2. Navigate to API Keys
3. Create new key
4. Copy key (starts with `sk-ant-...`)

### Step 3: Configure Environment

```bash
# Agent configuration
cd agents
cp .env.example .env

# Edit .env
nano .env
```

Add your configuration:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here

# Optional (defaults work for local development)
NATS_URL=nats://localhost:4222
MONGODB_URI=mongodb://localhost:27017/qaero
REDIS_URL=redis://localhost:6379
ML_SERVICE_URL=http://localhost:8000
PHYSICS_SERVICE_URL=http://localhost:8001
QUANTUM_SERVICE_URL=http://localhost:8002
```

### Step 4: Install Python Dependencies

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r agents/requirements.txt

# Verify installation
python -c "import anthropic; print('✓ Anthropic SDK installed')"
python -c "import langgraph; print('✓ LangGraph installed')"
python -c "import nats; print('✓ NATS client installed')"
```

### Step 5: Install Frontend Dependencies

```bash
cd frontend
npm install

# Verify
npm list react react-markdown
```

---

## 🤖 Agent Deployment

### Option A: Docker Compose (Recommended for Production)

```bash
# Build all agent images
docker-compose -f docker-compose.agents.yml build

# Start infrastructure services
docker-compose -f docker-compose.agents.yml up -d nats mongodb redis qdrant

# Wait for services to be ready (30 seconds)
sleep 30

# Start agents
docker-compose -f docker-compose.agents.yml up -d \
  master-orchestrator \
  ml-agent \
  quantum-agent \
  physics-agent \
  analysis-agent

# View logs
docker-compose -f docker-compose.agents.yml logs -f master-orchestrator

# Check status
docker-compose -f docker-compose.agents.yml ps
```

### Option B: Local Development (Recommended for Testing)

```bash
# Terminal 1: Start NATS
docker-compose -f docker-compose.agents.yml up nats

# Terminal 2: Master Orchestrator
cd agents
source venv/bin/activate
python -m master_orchestrator.agent

# Terminal 3: ML Surrogate Agent
cd agents
source venv/bin/activate
python -m ml_surrogate.agent

# Terminal 4: (Optional) Other agents
cd agents
source venv/bin/activate
python -m quantum_optimizer.agent
```

---

## 🎨 Frontend Integration

### Step 1: Add Claude Chat to Frontend

```bash
cd frontend/src

# Files already created:
# - components/claude/ClaudeChat.tsx
# - components/claude/ClaudeChat.css
# - hooks/useAnthropic.ts
```

### Step 2: Update App.tsx

```typescript
// frontend/src/App.tsx
import React from 'react';
import { ClaudeChat } from './components/claude/ClaudeChat';

function App() {
  return (
    <div className="App">
      <div className="claude-chat-panel">
        <ClaudeChat
          initialContext={{
            mesh_id: "wing_v3.2",
            parameters: { velocity: 250, yaw: 0 }
          }}
        />
      </div>
    </div>
  );
}

export default App;
```

### Step 3: Add Backend Proxy (Backend API)

```javascript
// services/backend/routes/claude.js
const express = require('express');
const router = express.Router();
const fetch = require('node-fetch');

// Stream endpoint
router.post('/stream', async (req, res) => {
  const { messages, context } = req.body;

  // Set up SSE
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  try {
    // Forward to Master Orchestrator Agent
    const response = await fetch('http://master-orchestrator:6001/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages, context })
    });

    // Stream response to client
    response.body.on('data', (chunk) => {
      res.write(`data: ${chunk}\n\n`);
    });

    response.body.on('end', () => {
      res.write('data: [DONE]\n\n');
      res.end();
    });

  } catch (error) {
    res.write(`data: ${JSON.stringify({ type: 'error', error: error.message })}\n\n`);
    res.end();
  }
});

module.exports = router;
```

### Step 4: Start Frontend

```bash
cd frontend
npm run dev

# Access at http://localhost:3000
```

---

## ✅ Verification & Testing

### 1. Check Infrastructure Health

```bash
# NATS
curl http://localhost:8222/healthz
# Expected: ok

# MongoDB
docker exec qaero-mongodb mongosh --eval "db.adminCommand('ping')"
# Expected: { ok: 1 }

# Redis
docker exec qaero-redis redis-cli ping
# Expected: PONG

# Qdrant
curl http://localhost:6333/collections
# Expected: {"result": {...}}
```

### 2. Test Agent Communication

```bash
# Install NATS CLI
brew install nats-io/nats-tools/nats
# or
wget https://github.com/nats-io/natscli/releases/download/v0.1.1/nats-0.1.1-linux-amd64.tar.gz
tar -xzf nats-0.1.1-linux-amd64.tar.gz
sudo mv nats /usr/local/bin/

# Test NATS connectivity
nats account info --server=nats://localhost:4222

# Subscribe to agent events
nats sub "events.>" --server=nats://localhost:4222

# In another terminal, publish test event
nats pub events.test '{"message": "Hello from NATS"}' --server=nats://localhost:4222
```

### 3. Test Master Orchestrator

```bash
# Health check
curl http://localhost:6001/health

# Test query (requires backend endpoint)
curl -X POST http://localhost:4000/api/claude/message \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the L/D ratio?"}
    ],
    "context": {}
  }'
```

### 4. Test ML Agent via NATS

```bash
# Request prediction
nats req agent.ml.predict '{
  "mesh_id": "wing_test",
  "parameters": {"velocity": 250, "yaw": 0},
  "model_preference": "auto"
}' --server=nats://localhost:4222 --timeout=10s
```

### 5. Frontend Chat Test

1. Open http://localhost:3000
2. You should see the Claude Chat interface
3. Try: "Explain what downforce is"
4. Verify streaming response appears

---

## 📊 Monitoring Setup

### 1. Access Grafana

```bash
# Open Grafana
open http://localhost:3001

# Login: admin / admin (change on first login)
```

### 2. Configure Data Sources

Grafana → Configuration → Data Sources → Add:

1. **Prometheus**
   - URL: `http://prometheus:9090`
   - Click "Save & Test"

### 3. Import Dashboards

Dashboards are pre-configured in `agents/observability/grafana-dashboards/`

Or import manually:
- Go to Dashboards → Import
- Upload JSON files from `agents/observability/grafana-dashboards/`

### 4. View Metrics

Key dashboards:
- **Agent Performance** - Request latency, throughput
- **Claude API Usage** - Token consumption, cost tracking
- **NATS Messaging** - Message queue health
- **System Health** - CPU, memory, GPU usage

---

## 🐛 Troubleshooting

### Issue: Agent containers exit immediately

```bash
# Check logs
docker-compose -f docker-compose.agents.yml logs master-orchestrator

# Common causes:
# 1. Missing ANTHROPIC_API_KEY
docker-compose -f docker-compose.agents.yml exec master-orchestrator env | grep ANTHROPIC

# 2. NATS not ready
docker-compose -f docker-compose.agents.yml ps nats

# Solution: Restart agents after NATS is healthy
docker-compose -f docker-compose.agents.yml restart master-orchestrator
```

### Issue: "Claude API Error: 401 Unauthorized"

```bash
# Verify API key is set correctly
echo $ANTHROPIC_API_KEY

# Test API key directly
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-sonnet-4.5-20250929",
    "max_tokens": 10,
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

### Issue: NATS connection refused

```bash
# Ensure NATS is running
docker-compose -f docker-compose.agents.yml up -d nats

# Check NATS logs
docker-compose -f docker-compose.agents.yml logs nats

# Test connection
telnet localhost 4222
```

### Issue: Frontend can't connect to backend

```bash
# Check backend is running
curl http://localhost:4000/health

# Verify CORS settings in backend
# services/backend/app.js should have:
app.use(cors({
  origin: 'http://localhost:3000',
  credentials: true
}));
```

### Issue: GPU not detected in ML service

```bash
# Verify NVIDIA drivers
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi

# Restart Docker with NVIDIA runtime
sudo systemctl restart docker
```

---

## 🎓 Next Steps

1. **Read Documentation**
   - `Quantum Aero F1 Prototype GENAI Claude BOOSTED.md` - Complete GenAI architecture
   - `Q-Aero Integration.md` - Full system integration guide

2. **Try Example Queries**
   - "Optimize this wing for maximum downforce"
   - "Analyze pressure distribution on the front wing"
   - "Compare this design with baseline"

3. **Implement Additional Agents**
   - Follow patterns in `agents/master_orchestrator/` and `agents/ml_surrogate/`
   - Add to `docker-compose.agents.yml`

4. **Scale for Production**
   - Increase agent replicas in docker-compose
   - Set up Kubernetes (see `Quantum Aero F1 Prototype GENAI Claude BOOSTED.md`)
   - Configure load balancing

5. **Monitor & Optimize**
   - Track token usage in Grafana
   - Optimize prompts for cost
   - Implement caching strategies

---

## 📞 Support

- **Issues:** https://github.com/your-org/Quantum-Aero-F1-Prototype/issues
- **Documentation:** See project README and docs/
- **Claude API:** https://docs.anthropic.com/claude/docs

---

**🎉 Congratulations! Your AI-native aerodynamic optimization platform is ready!**
