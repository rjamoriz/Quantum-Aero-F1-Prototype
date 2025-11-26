"""
Diffusion Model API
FastAPI service for generative aerodynamic design
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import numpy as np

from .generator import AeroGeometryGenerator

# Initialize FastAPI app
app = FastAPI(title="Diffusion Generative Design API", version="1.0.0")

# Global generator instance
generator: Optional[AeroGeometryGenerator] = None


# Request/Response Models
class GenerateRequest(BaseModel):
    cl: float = 2.8
    cd: float = 0.4
    cm: float = -0.1
    volume: float = 0.5
    thickness: float = 0.12
    camber: float = 0.04
    span: float = 2.0
    chord: float = 1.0
    num_inference_steps: int = 50
    guidance_scale: float = 7.5


class GenerateResponse(BaseModel):
    shape: List[int]
    generation_time_s: float
    target_met: bool
    parameters: Dict[str, float]
    num_inference_steps: int
    guidance_scale: float


class OptimizeRequest(BaseModel):
    target_cl: float = 2.8
    target_cd: float = 0.4
    num_candidates: int = 100
    num_inference_steps: int = 25


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize generator on startup"""
    global generator
    
    generator = AeroGeometryGenerator()
    print("✓ Diffusion generator initialized")


@app.get("/")
async def root():
    """API root"""
    return {
        "service": "Diffusion Generative Design API",
        "version": "1.0.0",
        "status": "ready",
        "target": "5-second generation, 1000+ candidates/day"
    }


@app.post("/api/ml/diffusion/generate", response_model=GenerateResponse)
async def generate_geometry(request: GenerateRequest):
    """
    Generate 3D aerodynamic geometry
    
    Target: <5 seconds generation time
    """
    if generator is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")
    
    try:
        result = generator.generate(
            cl=request.cl,
            cd=request.cd,
            cm=request.cm,
            volume=request.volume,
            thickness=request.thickness,
            camber=request.camber,
            span=request.span,
            chord=request.chord,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            return_timing=True
        )
        
        return GenerateResponse(
            shape=list(result['shape']),
            generation_time_s=result['generation_time_s'],
            target_met=result['target_met'],
            parameters=result['parameters'],
            num_inference_steps=result['num_inference_steps'],
            guidance_scale=result['guidance_scale']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/diffusion/optimize")
async def optimize_design(request: OptimizeRequest):
    """
    Generate and optimize multiple candidates
    
    Target: 1000+ candidates/day
    """
    if generator is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")
    
    try:
        candidates = generator.optimize_for_targets(
            target_cl=request.target_cl,
            target_cd=request.target_cd,
            num_candidates=request.num_candidates,
            num_inference_steps=request.num_inference_steps
        )
        
        # Return top 10 candidates
        top_candidates = candidates[:10]
        
        return {
            'num_generated': len(candidates),
            'target_cl': request.target_cl,
            'target_cd': request.target_cd,
            'top_candidates': [
                {
                    'candidate_id': c['candidate_id'],
                    'quality_score': c['quality_score'],
                    'parameters': c['parameters'],
                    'shape': list(c['shape'])
                }
                for c in top_candidates
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/diffusion/capabilities")
async def get_capabilities():
    """Get generator capabilities"""
    return {
        'model': 'Diffusion U-Net',
        'resolution': '64x64x64',
        'generation_time_target': '5 seconds',
        'candidates_per_day_target': 1000,
        'conditioning': [
            'lift_coefficient',
            'drag_coefficient',
            'moment_coefficient',
            'volume',
            'thickness',
            'camber',
            'span',
            'chord'
        ],
        'export_formats': ['stl', 'step', 'iges'],
        'guidance_scale_range': [1.0, 15.0],
        'inference_steps_range': [10, 100]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
