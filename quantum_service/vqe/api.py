"""
VQE API Endpoints
FastAPI service for quantum-enhanced optimization
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import numpy as np

from .optimizer import VQEAeroOptimizer, create_aerodynamic_qubo

# Initialize FastAPI app
app = FastAPI(title="VQE Quantum Optimizer API", version="1.0.0")

# Global optimizer instance
optimizer: Optional[VQEAeroOptimizer] = None


# Request/Response Models
class OptimizeRequest(BaseModel):
    qubo_matrix: List[List[float]]
    warm_start_solution: Optional[List[int]] = None
    num_layers: int = 3
    num_qubits: int = 20


class OptimizeResponse(BaseModel):
    solution: List[int]
    energy: float
    num_iterations: int
    optimization_time: float
    converged: bool
    num_qubits: int
    circuit_depth: int
    backend: str


class HardwareStatusResponse(BaseModel):
    available: bool
    backend: str
    queue_length: int
    num_qubits: int
    error_rate: float


class AeroOptimizeRequest(BaseModel):
    num_variables: int = 20
    target_cl: float = 2.8
    target_cd: float = 0.4
    warm_start: Optional[List[int]] = None


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize VQE optimizer on startup"""
    global optimizer
    
    optimizer = VQEAeroOptimizer(
        num_qubits=20,
        optimizer='COBYLA',
        max_iterations=1000,
        use_hardware=False
    )
    print("✓ VQE optimizer initialized")


@app.get("/")
async def root():
    """API root"""
    return {
        "service": "VQE Quantum Optimizer API",
        "version": "1.0.0",
        "status": "ready",
        "qubits": "50-100",
        "backend": "IBM Quantum System One (simulator)"
    }


@app.post("/api/quantum/vqe/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest):
    """
    Run VQE optimization on QUBO problem
    
    Target: 50-100 qubit optimization
    """
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")
    
    try:
        # Convert to numpy
        qubo_matrix = np.array(request.qubo_matrix, dtype=np.float32)
        warm_start = np.array(request.warm_start_solution) if request.warm_start_solution else None
        
        # Optimize
        result = optimizer.optimize(
            qubo_matrix=qubo_matrix,
            warm_start_solution=warm_start,
            num_layers=request.num_layers
        )
        
        return OptimizeResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quantum/vqe/optimize-aero")
async def optimize_aerodynamic(request: AeroOptimizeRequest):
    """
    Optimize aerodynamic design using VQE
    
    Automatically creates QUBO from aerodynamic objectives
    """
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")
    
    try:
        # Create aerodynamic QUBO
        qubo = create_aerodynamic_qubo(
            num_variables=request.num_variables,
            target_cl=request.target_cl,
            target_cd=request.target_cd
        )
        
        # Optimize
        warm_start = np.array(request.warm_start) if request.warm_start else None
        result = optimizer.optimize(qubo, warm_start_solution=warm_start)
        
        # Add aerodynamic interpretation
        result['target_cl'] = request.target_cl
        result['target_cd'] = request.target_cd
        result['num_variables'] = request.num_variables
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/quantum/vqe/hardware-status", response_model=HardwareStatusResponse)
async def hardware_status():
    """
    Get IBM Quantum hardware status
    
    Returns availability and queue information
    """
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")
    
    try:
        status = optimizer.get_hardware_status()
        return HardwareStatusResponse(**status)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quantum/vqe/warm-start")
async def create_warm_start(ml_prediction: List[float]):
    """
    Create VQE warm-start parameters from ML prediction
    
    Args:
        ml_prediction: ML-predicted binary solution
    
    Returns:
        Initial parameters for VQE ansatz
    """
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")
    
    try:
        prediction = np.array(ml_prediction)
        params = optimizer.warm_start(prediction)
        
        return {
            'initial_parameters': params.tolist(),
            'num_parameters': len(params),
            'strategy': 'ml_guided'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/quantum/vqe/circuit-metrics")
async def circuit_metrics(num_qubits: int = 20, num_layers: int = 3):
    """
    Get quantum circuit metrics
    
    Returns circuit depth, gate count, etc.
    """
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")
    
    try:
        # Calculate metrics
        num_rotation_gates = num_qubits * num_layers
        num_cnot_gates = (num_qubits - 1 + 1) * num_layers  # Linear + wrap-around
        total_gates = num_rotation_gates + num_cnot_gates
        circuit_depth = num_layers * 2  # Rotation + entanglement per layer
        
        return {
            'num_qubits': num_qubits,
            'num_layers': num_layers,
            'num_rotation_gates': num_rotation_gates,
            'num_cnot_gates': num_cnot_gates,
            'total_gates': total_gates,
            'circuit_depth': circuit_depth,
            'num_parameters': num_qubits * num_layers
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
