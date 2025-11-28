# Quantum-Aero F1 Multi-Agent System

**AI-Native Aerodynamic Optimization powered by Claude AI and Agentcy Framework**

This directory contains the implementation of the multi-agent system that powers the Q-Aero platform with intelligent, conversational aerodynamic optimization capabilities.

---

## 🎯 Overview

The Q-Aero multi-agent system consists of **8 specialized Claude AI agents** that work together to solve complex aerodynamic optimization problems:

1. **Master Orchestrator** - Coordinates all agents and maintains conversation context
2. **Intent Router** - Routes tasks to appropriate specialized agents
3. **Aerodynamics Agent** - CFD simulation analysis and interpretation
4. **ML Surrogate Agent** - Fast ML-based predictions with confidence assessment
5. **Quantum Optimizer Agent** - Quantum-enhanced design optimization
6. **Physics Validator Agent** - Physics-based validation of ML predictions
7. **Visualization Agent** - Result visualization and insight extraction
8. **Analysis Agent** - Trade-off analysis and recommendations

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│               (React + Claude Chat)                      │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│            Master Orchestrator Agent                     │
│              (Claude Sonnet 4.5)                         │
└──────────────────┬──────────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │   NATS Pub-Sub    │
         │   SLIM Transport  │
         └─────────┬─────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
┌────────┐   ┌─────────┐   ┌──────────┐
│ML Agent│   │Physics  │   │Quantum   │
│(Haiku) │   │Agent    │   │Agent     │
└────────┘   └─────────┘   └──────────┘
    │              │              │
    └──────────────┼──────────────┘
                   ▼
         ┌─────────────────┐
         │  ML/Physics/    │
         │  Quantum        │
         │  Services       │
         └─────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Anthropic API key
- NATS server (included in docker-compose)

### 2. Installation

```bash
# Install Python dependencies
cd agents
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Start Infrastructure

```bash
# Start NATS, databases, and observability stack
docker-compose -f docker-compose.agents.yml up -d nats mongodb redis qdrant prometheus grafana
```

### 4. Run Agents Locally (Development)

```bash
# Terminal 1: Master Orchestrator
python -m agents.master_orchestrator.agent

# Terminal 2: ML Surrogate Agent
python -m agents.ml_surrogate.agent

# Terminal 3: Add more agents as needed
```

### 5. Run Agents with Docker (Production)

```bash
# Build and start all agents
docker-compose -f docker-compose.agents.yml up --build

# Or start specific agents
docker-compose -f docker-compose.agents.yml up master-orchestrator ml-agent
```

---

## 📂 Directory Structure

```
agents/
├── config/
│   └── config.py                    # Centralized configuration
├── utils/
│   ├── anthropic_client.py          # Claude API client
│   └── nats_client.py               # NATS messaging client
├── master_orchestrator/
│   ├── agent.py                     # Master orchestrator implementation
│   └── Dockerfile
├── ml_surrogate/
│   ├── agent.py                     # ML surrogate implementation
│   └── Dockerfile
├── quantum_optimizer/
│   ├── agent.py                     # Quantum optimizer (to be implemented)
│   └── Dockerfile
├── physics_validator/
│   ├── agent.py                     # Physics validator (to be implemented)
│   └── Dockerfile
├── analysis/
│   ├── agent.py                     # Analysis agent (to be implemented)
│   └── Dockerfile
├── mcp_servers/
│   ├── mesh_database_server.py      # MCP server for mesh access
│   └── simulation_history_server.py # MCP server for simulation history
├── workflows/
│   └── optimization_workflow.py     # LangGraph workflows
├── observability/
│   ├── otel-collector-config.yaml   # OpenTelemetry config
│   ├── prometheus.yml               # Prometheus config
│   └── grafana-dashboards/          # Grafana dashboards
├── requirements.txt
├── .env.example
└── README.md (this file)
```

---

## 🔧 Configuration

### Environment Variables

See `.env.example` for all available configuration options:

```bash
# Core
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Infrastructure
NATS_URL=nats://localhost:4222
MONGODB_URI=mongodb://localhost:27017/qaero
REDIS_URL=redis://localhost:6379

# Services
ML_SERVICE_URL=http://localhost:8000
PHYSICS_SERVICE_URL=http://localhost:8001
QUANTUM_SERVICE_URL=http://localhost:8002
```

### Agent Configuration

Agent-specific configuration is in `config/config.py`:

```python
AGENT_MODELS = {
    "master_orchestrator": {
        "model": "claude-sonnet-4.5-20250929",
        "temperature": 0.2,
        "max_tokens": 4096,
    },
    "ml_surrogate": {
        "model": "claude-haiku-4-20250514",  # Fast & cost-efficient
        "temperature": 0.1,
        "max_tokens": 2048,
    },
    # ... more agents
}
```

---

## 💬 Usage Examples

### Example 1: Query Master Orchestrator

```python
from agents.master_orchestrator.agent import MasterOrchestratorAgent

agent = MasterOrchestratorAgent()
await agent.start()

# Stream response
async for chunk in agent.process_query(
    "Optimize this wing for maximum downforce at Monza",
    context={"mesh_id": "wing_v3.2", "velocity": 250}
):
    print(chunk, end="", flush=True)
```

### Example 2: Request ML Prediction via NATS

```python
from agents.utils.nats_client import NATSClient

nats = NATSClient()
await nats.connect()

response = await nats.request(
    "agent.ml.predict",
    {
        "mesh_id": "wing_v3.2",
        "parameters": {"velocity": 250, "yaw": 0},
        "model_preference": "auto"
    },
    timeout=10.0
)

print(f"Prediction: Cl={response['prediction']['Cl']:.2f}")
print(f"Confidence: {response['confidence']:.2f}")
```

### Example 3: Full Optimization Workflow

```python
from agents.workflows.optimization_workflow import run_optimization_workflow

result = await run_optimization_workflow(
    user_query="Optimize for Monza high-speed track",
    mesh_id="wing_v6.3"
)

print(f"Optimal config: {result['optimal_config']}")
print(f"Expected downforce gain: {result['analysis']['improvements']['downforce_gain']}%")
```

---

## 🧪 Testing

### Unit Tests

```bash
pytest agents/tests/
```

### Integration Tests

```bash
# Ensure NATS is running
docker-compose -f docker-compose.agents.yml up -d nats

# Run integration tests
pytest agents/tests/integration/
```

### Manual Testing with NATS CLI

```bash
# Install NATS CLI
brew install nats-io/nats-tools/nats

# Subscribe to agent events
nats sub "events.>"

# Publish test request
nats req agent.ml.predict '{"mesh_id": "wing_test", "parameters": {"velocity": 250}}'
```

---

## 📊 Monitoring & Observability

### Prometheus Metrics

Access at: `http://localhost:9090`

Key metrics:
- `agent_requests_total` - Total requests per agent
- `agent_request_duration_seconds` - Request latency
- `agent_errors_total` - Error count
- `claude_api_calls_total` - Claude API usage
- `claude_tokens_used_total` - Token consumption

### Grafana Dashboards

Access at: `http://localhost:3001` (default: admin/admin)

Pre-configured dashboards:
1. **Agent Performance** - Request latency, throughput, errors
2. **Claude API Usage** - API calls, token usage, cost tracking
3. **NATS Messaging** - Message queue depth, pub-sub latency
4. **System Health** - Agent status, resource usage

### OpenTelemetry Tracing

All agent interactions are traced with OpenTelemetry:

```python
from agents.observability.tracer import tracer

@tracer.start_as_current_span("ml_agent_inference")
async def predict(mesh_id, parameters):
    # Automatically traced
    result = await ml_service.predict(mesh_id, parameters)
    return result
```

View traces in Grafana with Tempo or Jaeger integration.

---

## 💰 Cost Optimization

### Token Usage Estimates

| Agent | Model | Tokens/Request | Cost/Request | Usage Pattern |
|-------|-------|----------------|--------------|---------------|
| Master Orchestrator | Sonnet 4.5 | ~2,000 | $0.012 | Every user query |
| ML Agent | Haiku | ~500 | $0.00125 | High frequency |
| Quantum Agent | Sonnet 4.5 | ~1,500 | $0.009 | Optimization only |
| Analysis Agent | Sonnet 4.5 | ~3,000 | $0.018 | Final synthesis |

**Estimated Monthly Cost:** ~$560 for 3,500 interactions/day

### Optimization Strategies

1. **Use Haiku for simple tasks** - ML agent uses Haiku for 60% cost savings
2. **Cache frequent queries** - Redis caching reduces redundant API calls
3. **Batch requests** - Process multiple predictions in one call
4. **Prompt compression** - Use concise system prompts
5. **Conditional agent calls** - Only invoke agents when necessary

---

## 🔐 Security

### API Key Management

```bash
# Use environment variables (never commit)
export ANTHROPIC_API_KEY=sk-ant-...

# Or use secret management
kubectl create secret generic anthropic-key \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-...
```

### NATS Authentication

```yaml
# nats.conf
authorization {
  users = [
    {user: "qaero", password: "$NATS_PASSWORD"}
  ]
}
```

### Network Security

- Agents communicate only via internal `qaero-agent-network`
- External access only through NGINX reverse proxy
- TLS/SSL for all external connections

---

## 🐛 Troubleshooting

### Agent Not Starting

```bash
# Check logs
docker-compose -f docker-compose.agents.yml logs master-orchestrator

# Verify NATS connection
docker-compose -f docker-compose.agents.yml exec nats nats server info

# Check environment variables
docker-compose -f docker-compose.agents.yml exec master-orchestrator env | grep ANTHROPIC
```

### Claude API Errors

```bash
# Rate limit exceeded
# Solution: Implement exponential backoff (already in anthropic_client.py)

# Invalid API key
# Solution: Verify ANTHROPIC_API_KEY in .env

# Context length exceeded
# Solution: Reduce conversation history (kept to last 20 messages)
```

### NATS Connection Issues

```bash
# Check NATS status
docker-compose -f docker-compose.agents.yml ps nats

# Test connection
nats account info --server=nats://localhost:4222

# Monitor messages
nats sub ">" --server=nats://localhost:4222
```

---

## 📚 Additional Resources

- **Main Documentation:** See `/Quantum Aero F1 Prototype GENAI Claude BOOSTED.md`
- **Agentcy Framework:** https://github.com/rjamoriz/Agentcy-Multiagents
- **Anthropic Claude:** https://docs.anthropic.com/claude/docs
- **LangGraph:** https://langchain-ai.github.io/langgraph/
- **NATS Messaging:** https://docs.nats.io/

---

## 🤝 Contributing

When adding new agents:

1. Create agent directory: `agents/{agent_name}/`
2. Implement `agent.py` with proper system prompt
3. Add Dockerfile following existing pattern
4. Update `config/config.py` with agent configuration
5. Add to `docker-compose.agents.yml`
6. Update this README

---

## 📝 License

Part of the Quantum-Aero F1 Prototype project.

---

**Built with ❤️ using Claude AI, Agentcy, and LangGraph**
