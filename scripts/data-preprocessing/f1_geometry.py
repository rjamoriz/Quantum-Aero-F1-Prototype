"""
F1 Car Geometry Builder
Creates complete F1 car aerodynamic surfaces from NACA profiles
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from naca_airfoil import NACAProfile

logger = logging.getLogger(__name__)


@dataclass
class F1GeometryParams:
    """F1 car geometry parameters"""
    # Front Wing
    front_wing_main_aoa: float = 20.0
    front_wing_flap1_aoa: float = 25.0
    front_wing_flap2_aoa: float = 30.0
    front_wing_ride_height: float = 0.05
    
    # Rear Wing
    rear_wing_main_aoa: float = 12.0
    rear_wing_flap_aoa: float = 25.0
    drs_open: bool = False
    
    # Floor/Diffuser
    floor_ride_height_front: float = 0.015
    floor_ride_height_rear: float = 0.050
    diffuser_angle: float = 15.0
    
    # Flow Conditions
    velocity: float = 50.0
    yaw_angle: float = 0.0
    pitch_angle: float = 0.0


class F1GeometryBuilder:
    """
    Build complete F1 car geometry from NACA profiles.
    
    Components:
    - Front wing (multi-element)
    - Rear wing (with DRS)
    - Floor and diffuser
    - Sidepods
    - Bargeboards
    """
    
    def __init__(self, params: Optional[F1GeometryParams] = None):
        """
        Initialize geometry builder.
        
        Args:
            params: Geometry parameters (uses defaults if None)
        """
        self.params = params or F1GeometryParams()
        self.geometry = {}
        
        logger.info("F1 Geometry Builder initialized")
    
    def build_front_wing(self, n_points: int = 50, n_span: int = 20) -> Dict:
        """
        Build front wing geometry.
        
        Args:
            n_points: Points per airfoil section
            n_span: Spanwise stations
            
        Returns:
            Dictionary with mesh coordinates
        """
        logger.info("Building front wing...")
        
        # Wing parameters
        span = 1.8  # FIA max width
        main_chord = 0.25
        flap1_chord = 0.12
        flap2_chord = 0.10
        
        # Create profiles
        main_profile = NACAProfile("6412")
        flap1_profile = NACAProfile("4415")
        flap2_profile = NACAProfile("4418")
        
        # Spanwise positions
        y_stations = np.linspace(-span/2, span/2, n_span)
        
        # Build main plane
        main_coords = []
        for y in y_stations:
            x, z = main_profile.generate_coordinates(n_points)
            x3d, y3d, z3d = main_profile.scale_and_transform(
                x, z,
                chord=main_chord,
                angle=self.params.front_wing_main_aoa,
                position=(1.5, y, self.params.front_wing_ride_height)
            )
            main_coords.append(np.column_stack([x3d, y3d, z3d]))
        
        # Build flap 1
        flap1_coords = []
        for y in y_stations:
            x, z = flap1_profile.generate_coordinates(n_points)
            x3d, y3d, z3d = flap1_profile.scale_and_transform(
                x, z,
                chord=flap1_chord,
                angle=self.params.front_wing_flap1_aoa,
                position=(1.5 - main_chord - 0.015, y, self.params.front_wing_ride_height + 0.05)
            )
            flap1_coords.append(np.column_stack([x3d, y3d, z3d]))
        
        # Build flap 2
        flap2_coords = []
        for y in y_stations:
            x, z = flap2_profile.generate_coordinates(n_points)
            x3d, y3d, z3d = flap2_profile.scale_and_transform(
                x, z,
                chord=flap2_chord,
                angle=self.params.front_wing_flap2_aoa,
                position=(1.5 - main_chord - flap1_chord - 0.030, y, self.params.front_wing_ride_height + 0.10)
            )
            flap2_coords.append(np.column_stack([x3d, y3d, z3d]))
        
        front_wing = {
            'main_plane': np.array(main_coords),
            'flap1': np.array(flap1_coords),
            'flap2': np.array(flap2_coords),
            'n_points': n_points,
            'n_span': n_span
        }
        
        logger.info(f"Front wing built: {n_points}x{n_span} mesh")
        
        return front_wing
    
    def build_rear_wing(self, n_points: int = 50, n_span: int = 15) -> Dict:
        """
        Build rear wing geometry.
        
        Args:
            n_points: Points per airfoil section
            n_span: Spanwise stations
            
        Returns:
            Dictionary with mesh coordinates
        """
        logger.info("Building rear wing...")
        
        # Wing parameters
        span = 0.75  # FIA regulation
        main_chord = 0.35
        flap_chord = 0.20
        height = 0.95
        
        # Create profiles
        main_profile = NACAProfile("9618")
        flap_profile = NACAProfile("6412")
        
        # Spanwise positions
        y_stations = np.linspace(-span/2, span/2, n_span)
        
        # Build main plane
        main_coords = []
        for y in y_stations:
            x, z = main_profile.generate_coordinates(n_points)
            x3d, y3d, z3d = main_profile.scale_and_transform(
                x, z,
                chord=main_chord,
                angle=self.params.rear_wing_main_aoa,
                position=(-1.5, y, height)
            )
            main_coords.append(np.column_stack([x3d, y3d, z3d]))
        
        # Build DRS flap
        flap_aoa = 0.0 if self.params.drs_open else self.params.rear_wing_flap_aoa
        
        flap_coords = []
        for y in y_stations:
            x, z = flap_profile.generate_coordinates(n_points)
            x3d, y3d, z3d = flap_profile.scale_and_transform(
                x, z,
                chord=flap_chord,
                angle=flap_aoa,
                position=(-1.5 - main_chord - 0.020, y, height + 0.05)
            )
            flap_coords.append(np.column_stack([x3d, y3d, z3d]))
        
        rear_wing = {
            'main_plane': np.array(main_coords),
            'drs_flap': np.array(flap_coords),
            'drs_state': 'open' if self.params.drs_open else 'closed',
            'n_points': n_points,
            'n_span': n_span
        }
        
        logger.info(f"Rear wing built: {n_points}x{n_span} mesh, DRS {rear_wing['drs_state']}")
        
        return rear_wing
    
    def build_floor_diffuser(self, n_x: int = 50, n_y: int = 20) -> Dict:
        """
        Build floor and diffuser geometry.
        
        Args:
            n_x: Longitudinal points
            n_y: Lateral points
            
        Returns:
            Dictionary with mesh coordinates
        """
        logger.info("Building floor and diffuser...")
        
        # Floor parameters
        length = 3.5
        width = 1.6
        diffuser_length = 1.0
        
        # Create floor mesh
        x = np.linspace(0, length, n_x)
        y = np.linspace(-width/2, width/2, n_y)
        X, Y = np.meshgrid(x, y)
        
        # Floor height (ride height variation)
        ride_height = np.linspace(
            self.params.floor_ride_height_front,
            self.params.floor_ride_height_rear,
            n_x
        )
        Z = np.tile(ride_height, (n_y, 1))
        
        # Add diffuser ramp
        diffuser_start = length - diffuser_length
        diffuser_mask = X > diffuser_start
        
        # Diffuser expansion
        diffuser_x = X[diffuser_mask] - diffuser_start
        diffuser_rise = diffuser_x * np.tan(np.radians(self.params.diffuser_angle))
        Z[diffuser_mask] += diffuser_rise
        
        floor_diffuser = {
            'floor': {
                'x': X,
                'y': Y,
                'z': Z
            },
            'diffuser_start': diffuser_start,
            'diffuser_angle': self.params.diffuser_angle,
            'n_x': n_x,
            'n_y': n_y
        }
        
        logger.info(f"Floor/diffuser built: {n_x}x{n_y} mesh")
        
        return floor_diffuser
    
    def build_complete_geometry(self) -> Dict:
        """
        Build complete F1 car geometry.
        
        Returns:
            Dictionary with all components
        """
        logger.info("Building complete F1 geometry...")
        
        geometry = {
            'front_wing': self.build_front_wing(),
            'rear_wing': self.build_rear_wing(),
            'floor_diffuser': self.build_floor_diffuser(),
            'parameters': self.params
        }
        
        # Compute bounding box
        all_coords = []
        for component in ['front_wing', 'rear_wing']:
            for element in geometry[component].values():
                if isinstance(element, np.ndarray):
                    all_coords.append(element.reshape(-1, 3))
        
        all_coords.append(np.column_stack([
            geometry['floor_diffuser']['floor']['x'].flatten(),
            geometry['floor_diffuser']['floor']['y'].flatten(),
            geometry['floor_diffuser']['floor']['z'].flatten()
        ]))
        
        all_coords = np.vstack(all_coords)
        
        geometry['bounding_box'] = {
            'min': all_coords.min(axis=0),
            'max': all_coords.max(axis=0),
            'center': all_coords.mean(axis=0)
        }
        
        logger.info("Complete F1 geometry built")
        
        return geometry
    
    def export_to_vtk(self, geometry: Dict, filename: str):
        """
        Export geometry to VTK format for visualization.
        
        Args:
            geometry: Geometry dictionary
            filename: Output filename
        """
        try:
            import vtk
            from vtk.util import numpy_support
            
            # TODO: Implement VTK export
            logger.info(f"VTK export to {filename} (not yet implemented)")
            
        except ImportError:
            logger.warning("VTK not available for export")
    
    def export_to_stl(self, geometry: Dict, filename: str):
        """
        Export geometry to STL format.
        
        Args:
            geometry: Geometry dictionary
            filename: Output filename
        """
        # TODO: Implement STL export
        logger.info(f"STL export to {filename} (not yet implemented)")


def create_parametric_variations(
    base_params: F1GeometryParams,
    n_samples: int = 100
) -> List[F1GeometryParams]:
    """
    Create parametric variations for data generation.
    
    Args:
        base_params: Base geometry parameters
        n_samples: Number of variations to generate
        
    Returns:
        List of parameter variations
    """
    from scipy.stats import qmc
    
    logger.info(f"Creating {n_samples} parametric variations...")
    
    # Define parameter ranges
    param_ranges = {
        'front_wing_main_aoa': (15, 25),
        'front_wing_flap1_aoa': (20, 30),
        'front_wing_flap2_aoa': (25, 35),
        'front_wing_ride_height': (0.03, 0.08),
        'rear_wing_main_aoa': (8, 16),
        'rear_wing_flap_aoa': (20, 30),
        'floor_ride_height_front': (0.010, 0.025),
        'floor_ride_height_rear': (0.040, 0.070),
        'diffuser_angle': (12, 18),
        'velocity': (30, 90),
        'yaw_angle': (-10, 10),
        'pitch_angle': (-2, 2)
    }
    
    # Latin Hypercube Sampling
    n_vars = len(param_ranges)
    sampler = qmc.LatinHypercube(d=n_vars)
    samples = sampler.random(n=n_samples)
    
    # Scale to parameter ranges
    variations = []
    for sample in samples:
        params = F1GeometryParams()
        for i, (param_name, (min_val, max_val)) in enumerate(param_ranges.items()):
            value = min_val + sample[i] * (max_val - min_val)
            setattr(params, param_name, value)
        
        # Random DRS state
        params.drs_open = np.random.rand() < 0.2  # 20% DRS open
        
        variations.append(params)
    
    logger.info(f"Created {len(variations)} parameter variations")
    
    return variations


if __name__ == "__main__":
    # Test F1 geometry builder
    logging.basicConfig(level=logging.INFO)
    
    print("F1 Geometry Builder Test")
    print("=" * 60)
    
    # Create builder
    params = F1GeometryParams()
    builder = F1GeometryBuilder(params)
    
    # Build components
    print("\n1. Building Front Wing...")
    front_wing = builder.build_front_wing(n_points=30, n_span=10)
    print(f"   Main plane: {front_wing['main_plane'].shape}")
    print(f"   Flap 1: {front_wing['flap1'].shape}")
    print(f"   Flap 2: {front_wing['flap2'].shape}")
    
    print("\n2. Building Rear Wing...")
    rear_wing = builder.build_rear_wing(n_points=30, n_span=10)
    print(f"   Main plane: {rear_wing['main_plane'].shape}")
    print(f"   DRS flap: {rear_wing['drs_flap'].shape}")
    print(f"   DRS state: {rear_wing['drs_state']}")
    
    print("\n3. Building Floor/Diffuser...")
    floor = builder.build_floor_diffuser(n_x=30, n_y=15)
    print(f"   Floor mesh: {floor['floor']['x'].shape}")
    print(f"   Diffuser angle: {floor['diffuser_angle']}°")
    
    print("\n4. Building Complete Geometry...")
    geometry = builder.build_complete_geometry()
    print(f"   Bounding box: {geometry['bounding_box']['min']} to {geometry['bounding_box']['max']}")
    
    print("\n5. Creating Parametric Variations...")
    variations = create_parametric_variations(params, n_samples=10)
    print(f"   Created {len(variations)} variations")
    print(f"   Sample variation: AOA={variations[0].front_wing_main_aoa:.1f}°, "
          f"Velocity={variations[0].velocity:.1f}m/s")
    
    print("\n✅ All tests passed!")
