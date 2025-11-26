"""
FastAPI server for Physics Engine Service
Provides VLM and Panel method aerodynamic calculations
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import sys
import os
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm.solver import VortexLatticeMethod, WingGeometry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Physics Engine API",
    description="Aerodynamic calculations using VLM and Panel methods",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class GeometryRequest(BaseModel):
    """Wing geometry parameters"""
    span: float = Field(..., gt=0, description="Wing span in meters")
    chord: float = Field(..., gt=0, description="Root chord in meters")
    twist: float = Field(0.0, description="Geometric twist in degrees")
    dihedral: float = Field(0.0, description="Dihedral angle in degrees")
    sweep: float = Field(0.0, description="Sweep angle in degrees")
    taper_ratio: float = Field(1.0, gt=0, le=1, description="Tip/root chord ratio")


class SimulationRequest(BaseModel):
    """VLM simulation request"""
    geometry: GeometryRequest
    velocity: float = Field(..., gt=0, description="Freestream velocity in m/s")
    alpha: float = Field(..., ge=-20, le=20, description="Angle of attack in degrees")
    yaw: float = Field(0.0, ge=-20, le=20, description="Yaw angle in degrees")
    rho: float = Field(1.225, gt=0, description="Air density in kg/m³")
    n_panels_x: int = Field(20, ge=5, le=100, description="Chordwise panels")
    n_panels_y: int = Field(10, ge=5, le=100, description="Spanwise panels")


class AeroResponse(BaseModel):
    """Aerodynamic results"""
    cl: float = Field(..., description="Lift coefficient")
    cd: float = Field(..., description="Drag coefficient")
    cm: float = Field(..., description="Moment coefficient")
    l_over_d: float = Field(..., description="Lift-to-drag ratio")
    lift: float = Field(..., description="Lift force in N")
    drag: float = Field(..., description="Drag force in N")
    side_force: float = Field(..., description="Side force in N")
    moment: float = Field(..., description="Pitching moment in N·m")
    pressure: List[float] = Field(..., description="Pressure coefficient distribution")
    n_panels: int = Field(..., description="Total number of panels")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str


# Global solver cache (for performance)
solver_cache = {}


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "Physics Engine API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="physics-engine",
        version="1.0.0"
    )


@app.post("/vlm/solve", response_model=AeroResponse)
async def solve_vlm(request: SimulationRequest):
    """
    Solve aerodynamics using Vortex Lattice Method.
    
    This endpoint computes aerodynamic forces and moments for a given
    wing geometry and flow conditions using the VLM solver.
    
    Args:
        request: Simulation parameters including geometry and flow conditions
        
    Returns:
        Aerodynamic coefficients, forces, and pressure distribution
        
    Raises:
        HTTPException: If solver fails or invalid parameters
    """
    try:
        logger.info(f"VLM solve request: V={request.velocity}m/s, α={request.alpha}°")
        
        # Create geometry
        geometry = WingGeometry(
            span=request.geometry.span,
            chord=request.geometry.chord,
            twist=request.geometry.twist,
            dihedral=request.geometry.dihedral,
            sweep=request.geometry.sweep,
            taper_ratio=request.geometry.taper_ratio
        )
        
        # Initialize solver
        cache_key = f"{request.n_panels_x}x{request.n_panels_y}"
        if cache_key not in solver_cache:
            solver_cache[cache_key] = VortexLatticeMethod(
                n_panels_x=request.n_panels_x,
                n_panels_y=request.n_panels_y
            )
        
        vlm = solver_cache[cache_key]
        vlm.setup_geometry(geometry)
        
        # Solve
        result = vlm.solve(
            velocity=request.velocity,
            alpha=request.alpha,
            yaw=request.yaw,
            rho=request.rho
        )
        
        # Prepare response
        l_over_d = result.cl / result.cd if result.cd > 0 else 0
        
        response = AeroResponse(
            cl=result.cl,
            cd=result.cd,
            cm=result.cm,
            l_over_d=l_over_d,
            lift=result.forces['lift'],
            drag=result.forces['drag'],
            side_force=result.forces['side'],
            moment=result.forces['moment'],
            pressure=result.pressure.tolist(),
            n_panels=request.n_panels_x * request.n_panels_y
        )
        
        logger.info(f"VLM solution: CL={result.cl:.4f}, CD={result.cd:.4f}")
        
        return response
        
    except Exception as e:
        logger.error(f"VLM solver error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Solver error: {str(e)}")


@app.post("/vlm/sweep")
async def alpha_sweep(
    geometry: GeometryRequest,
    velocity: float = Field(..., gt=0),
    alpha_start: float = Field(-10, ge=-20, le=20),
    alpha_end: float = Field(10, ge=-20, le=20),
    alpha_step: float = Field(1.0, gt=0, le=5),
    n_panels_x: int = Field(20, ge=5, le=100),
    n_panels_y: int = Field(10, ge=5, le=100)
):
    """
    Perform angle of attack sweep.
    
    Computes aerodynamic coefficients for a range of angles of attack.
    Useful for generating lift curves and finding optimal operating points.
    
    Args:
        geometry: Wing geometry
        velocity: Freestream velocity
        alpha_start: Starting angle of attack
        alpha_end: Ending angle of attack
        alpha_step: Step size
        n_panels_x: Chordwise panels
        n_panels_y: Spanwise panels
        
    Returns:
        List of results for each angle of attack
    """
    try:
        import numpy as np
        
        alphas = np.arange(alpha_start, alpha_end + alpha_step, alpha_step)
        results = []
        
        # Create geometry
        geom = WingGeometry(
            span=geometry.span,
            chord=geometry.chord,
            twist=geometry.twist,
            dihedral=geometry.dihedral,
            sweep=geometry.sweep,
            taper_ratio=geometry.taper_ratio
        )
        
        # Initialize solver
        vlm = VortexLatticeMethod(n_panels_x=n_panels_x, n_panels_y=n_panels_y)
        vlm.setup_geometry(geom)
        
        # Sweep through angles
        for alpha in alphas:
            result = vlm.solve(velocity=velocity, alpha=float(alpha))
            
            results.append({
                'alpha': float(alpha),
                'cl': result.cl,
                'cd': result.cd,
                'cm': result.cm,
                'l_over_d': result.cl / result.cd if result.cd > 0 else 0
            })
        
        logger.info(f"Alpha sweep complete: {len(results)} points")
        
        return {
            'sweep_type': 'alpha',
            'n_points': len(results),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Alpha sweep error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sweep error: {str(e)}")


@app.get("/vlm/validate")
async def validate_solver():
    """
    Validate VLM solver against known results.
    
    Tests the solver against NACA 0012 airfoil data at 5 degrees AoA.
    Expected CL ≈ 0.55 from experimental data.
    
    Returns:
        Validation results with error metrics
    """
    try:
        # NACA 0012 at 5 degrees
        geometry = WingGeometry(
            span=1.0,
            chord=0.2,
            twist=0.0,
            dihedral=0.0,
            sweep=0.0,
            taper_ratio=1.0
        )
        
        vlm = VortexLatticeMethod(n_panels_x=20, n_panels_y=10)
        vlm.setup_geometry(geometry)
        
        result = vlm.solve(velocity=50.0, alpha=5.0)
        
        # Expected values (from experimental data)
        expected_cl = 0.55
        expected_cd_range = (0.01, 0.05)
        
        # Compute errors
        cl_error = abs(result.cl - expected_cl) / expected_cl * 100
        cd_valid = expected_cd_range[0] <= result.cd <= expected_cd_range[1]
        
        validation = {
            'test': 'NACA 0012 at 5° AoA',
            'computed_cl': result.cl,
            'expected_cl': expected_cl,
            'cl_error_percent': cl_error,
            'computed_cd': result.cd,
            'cd_valid': cd_valid,
            'l_over_d': result.cl / result.cd,
            'passed': cl_error < 10 and cd_valid
        }
        
        logger.info(f"Validation: CL error = {cl_error:.2f}%")
        
        return validation
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
