# 🚀 Quick Start - GenAI Multi-Agent System

**Get up and running in 5 minutes!**

---

## ⚡ Fast Setup

### 1. Get Claude API Key (2 minutes)

```bash
# Visit: https://console.anthropic.com/
# Sign up → API Keys → Create Key
# Copy key: sk-ant-api03-...
```

### 2. Configure Environment (1 minute)

```bash
# Copy template
cp agents/.env.template agents/.env

# Edit .env and add your API key
nano agents/.env
# Set: ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE
```

### 3. Start Infrastructure (1 minute)

```bash
# Start NATS message broker
docker-compose -f docker-compose.agents.yml up -d nats

# Verify NATS is running
curl http://localhost:8222/varz
```

### 4. Deploy Agents (1 minute)

```bash
# Option A: Docker (recommended)
docker-compose -f docker-compose.agents.yml up -d

# Option B: Local development
source agents/venv/bin/activate
python agents/master_orchestrator/agent.py &
```

### 5. Test System (30 seconds)

```bash
# Check agent status
docker-compose -f docker-compose.agents.yml ps

# View logs
docker-compose -f docker-compose.agents.yml logs -f master-orchestrator
```

---

## 🎯 Usage Examples

### Example 1: Natural Language Query

```python
import asyncio
from agents.utils.nats_client import NATSClient

async def query_agent():
    nats = NATSClient()
    await nats.connect()
    
    response = await nats.request(
        "agent.orchestrator.query",
        {
            "query": "Optimize this wing for maximum downforce at Monza",
            "mesh_id": "wing_v3.2"
        }
    )
    
    print(response)
    await nats.disconnect()

asyncio.run(query_agent())
```

### Example 2: ML Prediction

```python
response = await nats.request(
    "agent.ml.predict",
    {
        "mesh_id": "wing_v3.2",
        "parameters": {"velocity": 250, "alpha": 5.0}
    }
)
```

### Example 3: Quantum Optimization

```python
response = await nats.request(
    "agent.quantum.optimize",
    {
        "objectives": ["maximize_downforce", "minimize_drag"],
        "constraints": {"flutter_margin": 1.2, "max_mass": 5.0}
    }
)
```

---

## 📊 Frontend Access

```bash
# Start frontend
cd frontend
npm install
npm start

# Open browser
open http://localhost:3000
```

### Available Components

1. **Claude Chat Interface** - Natural language queries
2. **Agent Activity Monitor** - Real-time agent status
3. **Agent Communication Graph** - Message flow visualization
4. **VLM Visualization** - Vortex lattice method
5. **Flow Field Visualization** - 3D flow rendering

---

## 🔍 Monitoring

### Check Agent Health

```bash
# All agents connected
curl http://localhost:8222/connz | jq '.connections[] | .name'

# NATS stats
curl http://localhost:8222/varz | jq
```

### View Logs

```bash
# All agents
docker-compose -f docker-compose.agents.yml logs -f

# Specific agent
docker logs qaero-master-orchestrator -f
```

### Metrics (Prometheus)

```bash
open http://localhost:9090
# Query: agent_requests_total
```

---

## 🛠️ Common Commands

### Start/Stop Agents

```bash
# Start all
docker-compose -f docker-compose.agents.yml up -d

# Stop all
docker-compose -f docker-compose.agents.yml down

# Restart specific agent
docker-compose -f docker-compose.agents.yml restart aerodynamics-agent
```

### Scale Agents

```bash
# Scale ML agent to 3 replicas
docker-compose -f docker-compose.agents.yml up -d --scale ml-agent=3
```

### Clean Up

```bash
# Stop and remove all
docker-compose -f docker-compose.agents.yml down -v

# Remove images
docker-compose -f docker-compose.agents.yml down --rmi all
```

---

## 🐛 Quick Troubleshooting

### Issue: Agent not starting

```bash
# Check logs
docker logs qaero-master-orchestrator

# Verify .env file
cat agents/.env | grep ANTHROPIC_API_KEY

# Restart agent
docker-compose -f docker-compose.agents.yml restart master-orchestrator
```

### Issue: NATS connection failed

```bash
# Check NATS is running
docker ps | grep nats

# Restart NATS
docker-compose -f docker-compose.agents.yml restart nats

# Check port
lsof -i :4222
```

### Issue: Claude API error

```bash
# Verify API key
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01"

# Check rate limits at console.anthropic.com
```

---

## 📚 Documentation

- **Full Setup Guide**: `GENAI_SETUP_GUIDE.md`
- **Implementation Plan**: `GENAI_IMPLEMENTATION_PLAN.md`
- **Project Status**: `FINAL_PROJECT_STATUS.md`
- **API Documentation**: `docs/API.md`

---

## 🎯 Next Steps

1. ✅ System running? → Test with example queries
2. ✅ Frontend working? → Try Claude Chat Interface
3. ✅ Agents responding? → Monitor communication graph
4. ✅ All good? → Start optimizing F1 wings!

---

## 💡 Pro Tips

- **Cache responses** to reduce API costs
- **Monitor logs** during development
- **Scale agents** for high load
- **Use mock data** for testing
- **Enable metrics** for observability

---

## 🆘 Need Help?

1. Check `GENAI_SETUP_GUIDE.md` for detailed instructions
2. Review agent logs for error messages
3. Verify NATS connection with health checks
4. Test Claude API key separately
5. Check Docker resources (memory, CPU)

---

**🎉 You're ready to use the GenAI Multi-Agent System!**

**Happy optimizing! 🏎️💨⚛️**
