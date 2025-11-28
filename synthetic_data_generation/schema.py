"""
Data schema for multi-fidelity synthetic aerodynamic dataset.
Defines the structure for F1-like configuration data across all tiers.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
from enum import Enum
import numpy as np
from datetime import datetime
import json


class FidelityTier(Enum):
    """Fidelity levels for synthetic data generation."""
    TIER_0_GEOMETRY = 0  # Parametric CAD/mesh descriptions
    TIER_1_FAST_PHYSICS = 1  # VLM / panel / lifting-line
    TIER_2_MEDIUM_FIDELITY = 2  # Unsteady VLM / vortex particle
    TIER_3_HIGH_FIDELITY = 3  # RANS / URANS / LES + FSI
    TIER_4_ML_AUGMENTED = 4  # ML-enriched augmentation


class SolverType(Enum):
    """Solver types used for generation."""
    VLM = "vlm"
    PANEL = "panel"
    LIFTING_LINE = "lifting_line"
    UNSTEADY_VLM = "unsteady_vlm"
    VORTEX_PARTICLE = "vortex_particle"
    RANS = "rans"
    URANS = "urans"
    LES = "les"
    FSI = "fsi"
    VAE = "vae"
    GAN = "gan"


@dataclass
class GeometryParameters:
    """Parametric description of F1 aerodynamic components."""
    
    # Front wing
    main_plane_chord: float  # cm
    main_plane_span: float  # cm
    main_plane_angle_deg: float  # -5 to 10
    flap1_angle_deg: float  # -5 to 20
    flap2_angle_deg: float  # -5 to 20
    endplate_height: float  # mm
    
    # Rear wing
    rear_wing_chord: float  # cm
    rear_wing_span: float  # cm
    rear_wing_angle_deg: float  # deg
    beam_wing_angle: float  # deg
    
    # Floor & diffuser
    floor_gap: float  # mm (ride height)
    diffuser_angle: float  # deg
    diffuser_length: float  # cm
    
    # Body
    sidepod_width: float  # cm
    sidepod_undercut: float  # cm
    
    # Control surfaces
    DRS_open: bool  # 0/1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML."""
        d = self.to_dict()
        d['DRS_open'] = float(d['DRS_open'])
        return np.array(list(d.values()), dtype=np.float32)
    
    @classmethod
    def get_bounds(cls) -> Dict[str, tuple]:
        """Get parameter bounds for sampling."""
        return {
            'main_plane_chord': (80.0, 120.0),
            'main_plane_span': (140.0, 180.0),
            'main_plane_angle_deg': (-5.0, 10.0),
            'flap1_angle_deg': (-5.0, 20.0),
            'flap2_angle_deg': (-5.0, 20.0),
            'endplate_height': (200.0, 400.0),
            'rear_wing_chord': (60.0, 100.0),
            'rear_wing_span': (80.0, 120.0),
            'rear_wing_angle_deg': (5.0, 25.0),
            'beam_wing_angle': (0.0, 15.0),
            'floor_gap': (10.0, 50.0),
            'diffuser_angle': (10.0, 25.0),
            'diffuser_length': (80.0, 150.0),
            'sidepod_width': (30.0, 60.0),
            'sidepod_undercut': (5.0, 20.0),
            'DRS_open': (0.0, 1.0),
        }


@dataclass
class FlowConditions:
    """Flow conditions for simulation."""
    V_inf: float  # m/s (freestream velocity)
    rho: float  # kg/m³ (air density)
    nu: float  # m²/s (kinematic viscosity)
    yaw: float  # deg (yaw angle)
    roll: float  # deg (roll angle)
    pitch: float  # deg (pitch angle)
    ground_gap: float  # mm (ground clearance)
    Re: float  # Reynolds number
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def standard_conditions(cls, V_inf: float = 70.0, ground_gap: float = 25.0) -> 'FlowConditions':
        """Create standard F1 flow conditions."""
        rho = 1.225  # kg/m³ at sea level
        nu = 1.5e-5  # m²/s for air
        L_ref = 1.0  # m (characteristic length)
        Re = V_inf * L_ref / nu
        
        return cls(
            V_inf=V_inf,
            rho=rho,
            nu=nu,
            yaw=0.0,
            roll=0.0,
            pitch=0.0,
            ground_gap=ground_gap,
            Re=Re
        )


@dataclass
class SimulationState:
    """State information for simulation."""
    is_steady: bool
    is_transient: bool
    time_step: Optional[float] = None  # seconds
    total_time: Optional[float] = None  # seconds
    n_timesteps: Optional[int] = None
    DRS_activation_time: Optional[float] = None  # seconds
    gust_profile: Optional[str] = None  # "none", "step", "ramp", "sine"


@dataclass
class GlobalOutputs:
    """Global aerodynamic outputs (scalars)."""
    CL: float  # Lift coefficient
    CD_total: float  # Total drag coefficient
    CD_induced: float  # Induced drag coefficient
    CD_pressure: Optional[float] = None  # Pressure drag
    CD_friction: Optional[float] = None  # Friction drag
    CM: float = 0.0  # Moment coefficient
    L: float = 0.0  # Lift force (N)
    D: float = 0.0  # Drag force (N)
    downforce_front: float = 0.0  # N
    downforce_rear: float = 0.0  # N
    balance: float = 0.0  # Front downforce / Total downforce
    L_over_D: float = 0.0  # Aerodynamic efficiency
    flutter_risk_estimate: Optional[float] = None  # 0-1 score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class FieldOutputs:
    """Field outputs (gridded or mesh-indexed data)."""
    Cp: Optional[np.ndarray] = None  # Pressure coefficient per panel
    Gamma: Optional[np.ndarray] = None  # Circulation per vortex/panel
    velocity_field: Optional[np.ndarray] = None  # (nx, ny, nz, 3)
    vorticity_field: Optional[np.ndarray] = None  # (nx, ny, nz, 3)
    pressure_field: Optional[np.ndarray] = None  # (nx, ny, nz)
    panel_centers: Optional[np.ndarray] = None  # (n_panels, 3)
    panel_normals: Optional[np.ndarray] = None  # (n_panels, 3)
    panel_areas: Optional[np.ndarray] = None  # (n_panels,)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (arrays as lists for JSON)."""
        d = {}
        for key, value in asdict(self).items():
            if value is not None and isinstance(value, np.ndarray):
                d[key] = {
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'data': 'stored_separately'  # Actual data in HDF5
                }
            else:
                d[key] = value
        return d


@dataclass
class TimeSeriesOutputs:
    """Time series outputs for transient simulations."""
    time: np.ndarray  # (n_timesteps,)
    CL_t: np.ndarray  # (n_timesteps,)
    CD_t: np.ndarray  # (n_timesteps,)
    CM_t: np.ndarray  # (n_timesteps,)
    max_displacement_t: Optional[np.ndarray] = None  # (n_timesteps,) for FSI
    flutter_amplitude_t: Optional[np.ndarray] = None  # (n_timesteps,)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'time_shape': self.time.shape,
            'n_timesteps': len(self.time),
            'time_range': (float(self.time[0]), float(self.time[-1])),
            'data': 'stored_separately'
        }


@dataclass
class ProvenanceMetadata:
    """Provenance and quality metadata."""
    solver: SolverType
    mesh_size: Optional[int] = None  # Number of cells/panels
    residuals: Optional[float] = None  # Final residual
    estimated_error: Optional[float] = None  # Error estimate
    runtime_seconds: Optional[float] = None
    convergence_achieved: bool = True
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['solver'] = self.solver.value
        return d


@dataclass
class AeroSample:
    """Complete aerodynamic sample (single simulation result)."""
    
    # Identifiers
    sample_id: str
    timestamp: str
    fidelity_tier: FidelityTier
    
    # Inputs
    geometry_params: GeometryParameters
    flow_conditions: FlowConditions
    state: SimulationState
    
    # Outputs
    global_outputs: GlobalOutputs
    field_outputs: Optional[FieldOutputs] = None
    timeseries_outputs: Optional[TimeSeriesOutputs] = None
    
    # Metadata
    provenance: ProvenanceMetadata = field(default_factory=lambda: ProvenanceMetadata(solver=SolverType.VLM))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'sample_id': self.sample_id,
            'timestamp': self.timestamp,
            'fidelity_tier': self.fidelity_tier.value,
            'geometry_params': self.geometry_params.to_dict(),
            'flow_conditions': self.flow_conditions.to_dict(),
            'state': asdict(self.state),
            'global_outputs': self.global_outputs.to_dict(),
            'field_outputs': self.field_outputs.to_dict() if self.field_outputs else None,
            'timeseries_outputs': self.timeseries_outputs.to_dict() if self.timeseries_outputs else None,
            'provenance': self.provenance.to_dict(),
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AeroSample':
        """Create from dictionary."""
        return cls(
            sample_id=data['sample_id'],
            timestamp=data['timestamp'],
            fidelity_tier=FidelityTier(data['fidelity_tier']),
            geometry_params=GeometryParameters(**data['geometry_params']),
            flow_conditions=FlowConditions(**data['flow_conditions']),
            state=SimulationState(**data['state']),
            global_outputs=GlobalOutputs(**data['global_outputs']),
            field_outputs=None,  # Load separately from HDF5
            timeseries_outputs=None,  # Load separately from HDF5
            provenance=ProvenanceMetadata(
                solver=SolverType(data['provenance']['solver']),
                **{k: v for k, v in data['provenance'].items() if k != 'solver'}
            )
        )


def generate_sample_id(tier: FidelityTier, index: int) -> str:
    """Generate unique sample ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"T{tier.value}_{timestamp}_{index:06d}"


if __name__ == "__main__":
    # Example usage
    print("=== Synthetic Aerodynamic Dataset Schema ===\n")
    
    # Create example geometry
    geom = GeometryParameters(
        main_plane_chord=100.0,
        main_plane_span=160.0,
        main_plane_angle_deg=5.0,
        flap1_angle_deg=10.0,
        flap2_angle_deg=15.0,
        endplate_height=300.0,
        rear_wing_chord=80.0,
        rear_wing_span=100.0,
        rear_wing_angle_deg=15.0,
        beam_wing_angle=8.0,
        floor_gap=25.0,
        diffuser_angle=18.0,
        diffuser_length=120.0,
        sidepod_width=45.0,
        sidepod_undercut=12.0,
        DRS_open=False
    )
    
    # Create flow conditions
    flow = FlowConditions.standard_conditions(V_inf=70.0)
    
    # Create state
    state = SimulationState(is_steady=True, is_transient=False)
    
    # Create outputs
    outputs = GlobalOutputs(
        CL=3.5,
        CD_total=0.85,
        CD_induced=0.65,
        L=15000.0,
        D=3500.0,
        downforce_front=6000.0,
        downforce_rear=9000.0,
        balance=0.4,
        L_over_D=4.29
    )
    
    # Create sample
    sample = AeroSample(
        sample_id=generate_sample_id(FidelityTier.TIER_1_FAST_PHYSICS, 0),
        timestamp=datetime.now().isoformat(),
        fidelity_tier=FidelityTier.TIER_1_FAST_PHYSICS,
        geometry_params=geom,
        flow_conditions=flow,
        state=state,
        global_outputs=outputs,
        provenance=ProvenanceMetadata(
            solver=SolverType.VLM,
            mesh_size=5000,
            runtime_seconds=2.5
        )
    )
    
    print("Sample JSON:")
    print(sample.to_json())
    
    print("\n=== Parameter Bounds ===")
    bounds = GeometryParameters.get_bounds()
    for param, (low, high) in bounds.items():
        print(f"{param:30s}: [{low:8.2f}, {high:8.2f}]")
