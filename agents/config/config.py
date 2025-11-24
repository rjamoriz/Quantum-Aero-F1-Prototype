"""
Q-Aero Agent Configuration
Centralized configuration for all Claude AI agents
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

# ============================================
# Anthropic Claude Configuration
# ============================================

ANTHROPIC_CONFIG = {
    "api_key": os.getenv("ANTHROPIC_API_KEY"),
    "default_model": "claude-sonnet-4.5-20250929",
    "haiku_model": "claude-haiku-4-20250514",
    "max_tokens": 4096,
    "temperature": 0.2,
}

# ============================================
# Agent Models Configuration
# ============================================

AGENT_MODELS = {
    "master_orchestrator": {
        "model": "claude-sonnet-4.5-20250929",
        "temperature": 0.2,
        "max_tokens": 4096,
    },
    "intent_router": {
        "model": "claude-haiku-4-20250514",
        "temperature": 0.1,
        "max_tokens": 1024,
    },
    "aerodynamics": {
        "model": "claude-sonnet-4.5-20250929",
        "temperature": 0.3,
        "max_tokens": 3072,
    },
    "ml_surrogate": {
        "model": "claude-haiku-4-20250514",
        "temperature": 0.1,
        "max_tokens": 2048,
    },
    "quantum_optimizer": {
        "model": "claude-sonnet-4.5-20250929",
        "temperature": 0.2,
        "max_tokens": 3072,
    },
    "physics_validator": {
        "model": "claude-sonnet-4.5-20250929",
        "temperature": 0.2,
        "max_tokens": 2048,
    },
    "analysis": {
        "model": "claude-sonnet-4.5-20250929",
        "temperature": 0.3,
        "max_tokens": 4096,
    },
    "visualization": {
        "model": "claude-haiku-4-20250514",
        "temperature": 0.1,
        "max_tokens": 1024,
    },
    "data_manager": {
        "model": "claude-haiku-4-20250514",
        "temperature": 0.1,
        "max_tokens": 1024,
    },
    "report_generator": {
        "model": "claude-sonnet-4.5-20250929",
        "temperature": 0.3,
        "max_tokens": 4096,
    },
}

# ============================================
# SLIM Transport Configuration
# ============================================

SLIM_CONFIG = {
    "transport": "slim",
    "host": os.getenv("SLIM_HOST", "localhost"),
    "protocol": "a2a",  # Agent-to-Agent protocol
    "patterns": {
        "request_reply": True,
        "unicast": True,
        "broadcast": True,
        "group": True,
    },
}

# Agent ports
AGENT_PORTS = {
    "master_orchestrator": 6001,
    "intent_router": 6002,
    "aerodynamics": 6003,
    "ml_surrogate": 6004,
    "quantum_optimizer": 6005,
    "physics_validator": 6006,
    "analysis": 6007,
    "visualization": 6008,
    "data_manager": 6009,
    "report_generator": 6010,
}

# ============================================
# NATS Configuration
# ============================================

NATS_CONFIG = {
    "servers": [os.getenv("NATS_URL", "nats://localhost:4222")],
    "max_reconnect_attempts": 10,
    "reconnect_time_wait": 2,
}

# NATS subjects
NATS_SUBJECTS = {
    "simulation_completed": "events.simulation.completed",
    "optimization_requested": "events.optimization.requested",
    "analysis_ready": "events.analysis.ready",
    "agent_broadcast": "group.agents.broadcast",
    "physics_validation": "group.physics.validate",
}

# ============================================
# Database Configuration
# ============================================

MONGODB_CONFIG = {
    "uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017/qaero"),
    "database": "qaero",
    "collections": {
        "meshes": "meshes",
        "simulations": "simulations",
        "conversations": "conversations",
        "agent_logs": "agent_logs",
    },
}

REDIS_CONFIG = {
    "url": os.getenv("REDIS_URL", "redis://localhost:6379"),
    "decode_responses": True,
    "max_connections": 50,
}

QDRANT_CONFIG = {
    "host": os.getenv("QDRANT_HOST", "localhost"),
    "port": int(os.getenv("QDRANT_PORT", "6333")),
    "collection": "qaero_embeddings",
}

# ============================================
# Service URLs
# ============================================

SERVICE_URLS = {
    "ml_inference": os.getenv("ML_SERVICE_URL", "http://localhost:8000"),
    "physics": os.getenv("PHYSICS_SERVICE_URL", "http://localhost:8001"),
    "quantum": os.getenv("QUANTUM_SERVICE_URL", "http://localhost:8002"),
    "fsi": os.getenv("FSI_SERVICE_URL", "http://localhost:8003"),
    "backend": os.getenv("BACKEND_URL", "http://localhost:4000"),
}

# ============================================
# OpenTelemetry Configuration
# ============================================

OTEL_CONFIG = {
    "endpoint": os.getenv("OTEL_ENDPOINT", "http://localhost:4317"),
    "service_name_prefix": "qaero-agent",
    "enabled": os.getenv("OTEL_ENABLED", "true").lower() == "true",
}

# ============================================
# Agent Behavior Configuration
# ============================================

AGENT_BEHAVIOR = {
    "request_timeout": 30.0,  # seconds
    "max_retries": 3,
    "retry_delay": 1.0,  # seconds
    "confidence_threshold": 0.9,
    "batch_size": 10,
}

# ============================================
# Safety Constraints
# ============================================

SAFETY_CONSTRAINTS = {
    "flutter_margin_min": 1.2,  # Vf > 1.2 × Vmax
    "stress_safety_factor": 1.5,  # σ_max < σ_yield / 1.5
    "max_wing_mass_kg": 5.0,
    "max_downforce_kN": 5.0,
    "min_L_D_ratio": 3.0,
}

# ============================================
# Cost Optimization
# ============================================

COST_CONFIG = {
    "enable_caching": True,
    "cache_ttl": 3600,  # seconds
    "use_haiku_for_simple_tasks": True,
    "batch_requests": True,
}


def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """Get configuration for a specific agent"""
    return {
        "anthropic": {
            **ANTHROPIC_CONFIG,
            **AGENT_MODELS.get(agent_name, {}),
        },
        "slim": {
            **SLIM_CONFIG,
            "port": AGENT_PORTS.get(agent_name, 6000),
        },
        "nats": NATS_CONFIG,
        "services": SERVICE_URLS,
        "behavior": AGENT_BEHAVIOR,
        "safety": SAFETY_CONSTRAINTS,
        "otel": {
            **OTEL_CONFIG,
            "service_name": f"{OTEL_CONFIG['service_name_prefix']}-{agent_name}",
        },
    }
