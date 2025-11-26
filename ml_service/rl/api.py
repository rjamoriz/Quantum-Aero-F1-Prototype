"""
RL Active Control API
FastAPI service for reinforcement learning-based aerodynamic control
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from .environment import F1AeroControlEnv
from .agent import PPOAgent

# Initialize FastAPI app
app = FastAPI(title="RL Active Control API", version="1.0.0")

# Global instances
env: Optional[F1AeroControlEnv] = None
agent: Optional[PPOAgent] = None


# Request/Response Models
class ControlRequest(BaseModel):
    velocity: float
    position: float
    track_curvature: float
    drs_available: bool = True


class ControlResponse(BaseModel):
    drs_activation: bool
    front_flap_angle: float
    rear_flap_angle: float
    predicted_cl: float
    predicted_cd: float
    confidence: float


class TrainRequest(BaseModel):
    num_episodes: int = 100
    update_interval: int = 2048
    save_path: str = "models/ppo_aero_control.pt"


class SimulateRequest(BaseModel):
    num_laps: int = 10
    use_trained_policy: bool = True


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize environment and agent on startup"""
    global env, agent
    
    env = F1AeroControlEnv()
    agent = PPOAgent(state_dim=10, action_dim=3, hidden_dim=256)
    
    # Try to load pretrained model
    try:
        agent.load("models/ppo_aero_control.pt")
        print("✓ Loaded pretrained RL agent")
    except:
        print("✓ RL agent initialized (no pretrained model)")


@app.get("/")
async def root():
    """API root"""
    return {
        "service": "RL Active Control API",
        "version": "1.0.0",
        "status": "ready",
        "description": "PPO-based DRS and flap optimization for F1"
    }


@app.post("/api/rl/control", response_model=ControlResponse)
async def get_control_action(request: ControlRequest):
    """
    Get optimal control action for current state
    
    Real-time inference for track deployment
    """
    if agent is None or env is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Construct state (simplified)
        state = np.array([
            request.velocity / 90.0,  # Normalized
            request.position,
            request.track_curvature,
            0.0,  # Next curvature (unknown)
            float(request.drs_available),
            0.0,  # Current front flap
            0.0,  # Current rear flap
            2.5 / 3.0,  # Current Cl
            0.45 / 1.0,  # Current Cd
            0.0  # Lap time
        ], dtype=np.float32)
        
        # Get action from policy
        action, _, _ = agent.select_action(state, deterministic=True)
        
        # Parse action
        drs_activation = action[0] > 0.5
        front_flap = float(action[1] * 15.0)  # Scale to degrees
        rear_flap = float(action[2] * 15.0)
        
        # Predict aerodynamics (simplified)
        cl = 2.5 + 0.02 * front_flap + 0.03 * rear_flap
        cd = 0.45 + 0.005 * abs(front_flap) + 0.008 * abs(rear_flap)
        
        if drs_activation:
            cl -= 0.3
            cd -= 0.15
        
        return ControlResponse(
            drs_activation=drs_activation,
            front_flap_angle=front_flap,
            rear_flap_angle=rear_flap,
            predicted_cl=cl,
            predicted_cd=cd,
            confidence=0.95
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rl/train")
async def train_agent(request: TrainRequest):
    """
    Train RL agent
    
    Background training on historical data or simulation
    """
    if agent is None or env is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Train agent
        stats = agent.train(
            env,
            num_episodes=request.num_episodes,
            update_interval=request.update_interval,
            save_path=request.save_path
        )
        
        return {
            'status': 'training_complete',
            'num_episodes': len(stats['episode_rewards']),
            'avg_reward': float(np.mean(stats['episode_rewards'][-10:])),
            'avg_lap_time': float(np.mean(stats['lap_times'][-10:])),
            'best_lap_time': float(np.min(stats['lap_times'])),
            'model_saved': request.save_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rl/simulate")
async def simulate_laps(request: SimulateRequest):
    """
    Simulate multiple laps with current policy
    
    Evaluate policy performance
    """
    if agent is None or env is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        lap_times = []
        avg_velocities = []
        
        for lap in range(request.num_laps):
            state = env.reset()
            done = False
            
            while not done:
                if request.use_trained_policy:
                    action, _, _ = agent.select_action(state, deterministic=True)
                else:
                    action = env.action_space.sample()
                
                state, _, done, info = env.step(action)
            
            lap_times.append(info['lap_time'])
            avg_velocities.append(env.track_length / info['lap_time'])
        
        return {
            'num_laps': request.num_laps,
            'lap_times': lap_times,
            'avg_lap_time': float(np.mean(lap_times)),
            'best_lap_time': float(np.min(lap_times)),
            'std_lap_time': float(np.std(lap_times)),
            'avg_velocity': float(np.mean(avg_velocities)),
            'policy_type': 'trained' if request.use_trained_policy else 'random'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/stats")
async def get_training_stats():
    """Get training statistics"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    stats = agent.training_stats
    
    if len(stats['episode_rewards']) == 0:
        return {
            'status': 'not_trained',
            'message': 'No training data available'
        }
    
    return {
        'status': 'trained',
        'total_episodes': len(stats['episode_rewards']),
        'avg_reward': float(np.mean(stats['episode_rewards'][-100:])),
        'avg_lap_time': float(np.mean(stats['lap_times'][-100:])),
        'best_lap_time': float(np.min(stats['lap_times'])),
        'recent_policy_loss': float(np.mean(stats['policy_losses'][-10:])) if stats['policy_losses'] else 0,
        'recent_value_loss': float(np.mean(stats['value_losses'][-10:])) if stats['value_losses'] else 0
    }


@app.get("/api/rl/capabilities")
async def get_capabilities():
    """Get RL agent capabilities"""
    return {
        'algorithm': 'PPO (Proximal Policy Optimization)',
        'state_space': {
            'velocity': 'normalized [0, 1]',
            'position': 'normalized [0, 1]',
            'track_curvature': '[-1, 1]',
            'drs_available': 'binary',
            'flap_angles': 'degrees [-15, 15]',
            'aerodynamics': 'Cl, Cd'
        },
        'action_space': {
            'drs_activation': 'binary [0, 1]',
            'front_flap_delta': 'degrees [-5, 5]',
            'rear_flap_delta': 'degrees [-5, 5]'
        },
        'optimization_target': 'lap_time_minimization',
        'constraints': ['safety', 'regulations', 'mechanical_limits'],
        'training': {
            'episodes': 1000,
            'update_interval': 2048,
            'batch_size': 64
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
