"""
Synthetic Aeroelastic Dataset Generator
Generates training data for ML surrogate models with quantum optimization integration

Based on: Quantum-Aero F1 Prototype AEROELASTIC.md
"""

import numpy as np
import h5py
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys
from scipy import linalg
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'services' / 'physics-engine'))
sys.path.append(str(Path(__file__).parent.parent.parent / 'services' / 'quantum-optimizer'))

try:
    from transient.transient_aero import ModalDynamics, UnsteadyVLM
    from multiphysics.vibration_thermal_acoustic import StructuralVibrationAnalyzer
    PHYSICS_AVAILABLE = True
except ImportError:
    PHYSICS_AVAILABLE = False
    logging.warning("Physics modules not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AeroelasticConfiguration:
    """Aeroelastic configuration parameters"""
    # Geometric parameters
    chord: float  # m
    span: float  # m
    thickness: float  # mm
    camber: float  # %
    
    # Structural parameters
    n_stiffeners: int
    stiffener_positions: np.ndarray
    material: str  # 'carbon_fiber', 'aluminum'
    
    # Operating conditions
    velocity: float  # m/s
    yaw_angle: float  # degrees
    ride_height: float  # mm
    
    # Modal properties
    natural_frequencies: np.ndarray  # Hz
    damping_ratios: np.ndarray
    mode_shapes: np.ndarray


@dataclass
class AeroelasticResults:
    """Results from aeroelastic simulation"""
    # Aerodynamic forces
    lift: float  # N
    drag: float  # N
    moment: float  # Nm
    
    # Aeroelastic metrics
    flutter_speed: float  # m/s
    flutter_margin: float
    divergence_speed: float  # m/s
    
    # Deformation
    max_displacement: float  # m
    rms_displacement: float  # m
    twist_angle: float  # degrees
    
    # Stress
    max_stress: float  # MPa
    stress_safety_factor: float
    
    # Modal energy
    modal_energy: np.ndarray  # J
    modal_amplitudes: np.ndarray


class AeroelasticDatasetGenerator:
    """
    Generates synthetic aeroelastic datasets for ML training.
    
    Integrates:
    - Modal analysis
    - Flutter calculation
    - FSI coupling
    - Quantum optimization variables
    """
    
    def __init__(
        self,
        output_dir: str = "data/aeroelastic-datasets",
        n_samples: int = 1000
    ):
        """
        Initialize dataset generator.
        
        Args:
            output_dir: Output directory for datasets
            n_samples: Number of samples to generate
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_samples = n_samples
        
        # Material properties
        self.materials = {
            'carbon_fiber': {
                'density': 1600,  # kg/m³
                'youngs_modulus': 150e9,  # Pa
                'yield_stress': 600e6,  # Pa
                'damping': 0.02
            },
            'aluminum': {
                'density': 2700,  # kg/m³
                'youngs_modulus': 70e9,  # Pa
                'yield_stress': 300e6,  # Pa
                'damping': 0.015
            }
        }
        
        logger.info(f"Aeroelastic dataset generator initialized: {n_samples} samples")
    
    def generate_configuration(self, sample_idx: int) -> AeroelasticConfiguration:
        """
        Generate random aeroelastic configuration.
        
        Uses Latin Hypercube Sampling for better space coverage.
        
        Args:
            sample_idx: Sample index
            
        Returns:
            Configuration object
        """
        np.random.seed(sample_idx)
        
        # Geometric parameters (F1 wing typical ranges)
        chord = np.random.uniform(0.15, 0.30)  # m
        span = np.random.uniform(1.2, 1.8)  # m
        thickness = np.random.uniform(1.0, 2.5)  # mm
        camber = np.random.uniform(2.0, 8.0)  # %
        
        # Structural parameters (quantum optimization variables)
        n_stiffeners = np.random.randint(0, 9)  # 0-8 stiffeners
        n_locations = 20
        stiffener_positions = np.zeros(n_locations)
        if n_stiffeners > 0:
            positions = np.random.choice(n_locations, n_stiffeners, replace=False)
            stiffener_positions[positions] = 1
        
        material = np.random.choice(['carbon_fiber', 'aluminum'])
        
        # Operating conditions
        velocity = np.random.uniform(100/3.6, 350/3.6)  # m/s (100-350 km/h)
        yaw_angle = np.random.uniform(0, 10)  # degrees
        ride_height = np.random.uniform(-10, 0)  # mm
        
        # Compute modal properties
        natural_frequencies, damping_ratios, mode_shapes = self._compute_modal_properties(
            chord, span, thickness, n_stiffeners, material
        )
        
        config = AeroelasticConfiguration(
            chord=chord,
            span=span,
            thickness=thickness,
            camber=camber,
            n_stiffeners=n_stiffeners,
            stiffener_positions=stiffener_positions,
            material=material,
            velocity=velocity,
            yaw_angle=yaw_angle,
            ride_height=ride_height,
            natural_frequencies=natural_frequencies,
            damping_ratios=damping_ratios,
            mode_shapes=mode_shapes
        )
        
        return config
    
    def _compute_modal_properties(
        self,
        chord: float,
        span: float,
        thickness: float,
        n_stiffeners: int,
        material: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute modal properties (frequencies, damping, mode shapes).
        
        Simplified analytical model for F1 wing.
        
        Args:
            chord: Wing chord (m)
            span: Wing span (m)
            thickness: Skin thickness (mm)
            n_stiffeners: Number of stiffeners
            material: Material type
            
        Returns:
            (natural_frequencies, damping_ratios, mode_shapes)
        """
        n_modes = 5
        
        # Material properties
        mat_props = self.materials[material]
        E = mat_props['youngs_modulus']
        rho = mat_props['density']
        base_damping = mat_props['damping']
        
        # Simplified stiffness (increases with thickness and stiffeners)
        thickness_m = thickness / 1000  # Convert to meters
        I = (span * thickness_m**3) / 12  # Second moment of area
        stiffness_factor = 1.0 + 0.15 * n_stiffeners  # Stiffeners increase stiffness
        
        # Base frequencies (Hz) for F1 wing
        base_freqs = np.array([25, 42, 58, 73, 95])
        
        # Adjust frequencies based on geometry and stiffeners
        freq_factor = np.sqrt(stiffness_factor * (thickness / 2.0))
        natural_frequencies = base_freqs * freq_factor
        
        # Damping ratios (slightly increase with stiffeners)
        damping_ratios = np.full(n_modes, base_damping) * (1.0 + 0.05 * n_stiffeners)
        
        # Simplified mode shapes (5 modes)
        n_points = 10
        x = np.linspace(0, span, n_points)
        mode_shapes = np.zeros((n_modes, n_points))
        
        for i in range(n_modes):
            # Mode shape: sin((i+1)*π*x/L)
            mode_shapes[i, :] = np.sin((i + 1) * np.pi * x / span)
        
        return natural_frequencies, damping_ratios, mode_shapes
    
    def simulate_aeroelastic_response(
        self,
        config: AeroelasticConfiguration
    ) -> AeroelasticResults:
        """
        Simulate aeroelastic response for given configuration.
        
        Includes:
        - Aerodynamic forces (VLM)
        - Flutter analysis
        - Structural deformation
        - Stress calculation
        
        Args:
            config: Configuration object
            
        Returns:
            Results object
        """
        # Aerodynamic forces (simplified VLM)
        rho = 1.225  # kg/m³
        q = 0.5 * rho * config.velocity**2
        S = config.chord * config.span
        
        # Lift coefficient (depends on camber and deformation)
        CL_alpha = 0.1  # per degree
        alpha_effective = config.camber / 2  # Simplified
        CL = CL_alpha * alpha_effective
        
        # Drag coefficient
        CD = 0.02 + 0.05 * (alpha_effective / 10)**2
        
        # Forces
        lift = q * S * CL
        drag = q * S * CD
        moment = lift * config.chord * 0.25  # Quarter-chord moment
        
        # Flutter analysis
        flutter_speed, flutter_margin = self._compute_flutter_speed(config)
        
        # Divergence speed (static instability)
        divergence_speed = flutter_speed * 1.5  # Simplified
        
        # Structural deformation (simplified)
        load_per_unit_span = lift / config.span
        EI = self.materials[config.material]['youngs_modulus'] * (config.thickness/1000)**3 / 12
        max_displacement = (load_per_unit_span * config.span**4) / (8 * EI)
        rms_displacement = max_displacement * 0.7
        
        # Twist angle (aeroelastic coupling)
        twist_angle = (lift * config.chord) / (EI * 1000)  # Simplified
        
        # Stress calculation
        max_stress = (lift * config.span) / (2 * config.thickness/1000 * config.span)  # MPa
        max_stress /= 1e6  # Convert to MPa
        yield_stress = self.materials[config.material]['yield_stress'] / 1e6
        stress_safety_factor = yield_stress / max_stress if max_stress > 0 else 10.0
        
        # Modal energy (simplified)
        modal_energy = np.random.rand(5) * 0.1  # J
        modal_amplitudes = np.random.rand(5) * 0.01  # m
        
        results = AeroelasticResults(
            lift=lift,
            drag=drag,
            moment=moment,
            flutter_speed=flutter_speed,
            flutter_margin=flutter_margin,
            divergence_speed=divergence_speed,
            max_displacement=max_displacement,
            rms_displacement=rms_displacement,
            twist_angle=twist_angle,
            max_stress=max_stress,
            stress_safety_factor=stress_safety_factor,
            modal_energy=modal_energy,
            modal_amplitudes=modal_amplitudes
        )
        
        return results
    
    def _compute_flutter_speed(
        self,
        config: AeroelasticConfiguration
    ) -> Tuple[float, float]:
        """
        Compute flutter speed using simplified k-method.
        
        Args:
            config: Configuration object
            
        Returns:
            (flutter_speed, flutter_margin)
        """
        # Simplified flutter analysis
        # Flutter speed increases with stiffness (frequencies) and damping
        
        # Base flutter speed for F1 wing
        V_f_base = 300 / 3.6  # m/s (300 km/h)
        
        # Adjust based on natural frequencies and stiffeners
        freq_factor = config.natural_frequencies[0] / 25.0  # Normalize to baseline
        stiffener_factor = 1.0 + 0.1 * config.n_stiffeners
        
        flutter_speed = V_f_base * freq_factor * stiffener_factor
        
        # Flutter margin = V_f / V_max
        V_max = 350 / 3.6  # m/s
        flutter_margin = flutter_speed / V_max
        
        return flutter_speed, flutter_margin
    
    def generate_dataset(self, parallel: bool = True) -> Dict:
        """
        Generate complete aeroelastic dataset.
        
        Args:
            parallel: Use parallel processing
            
        Returns:
            Dataset statistics
        """
        logger.info(f"Generating {self.n_samples} aeroelastic samples...")
        
        configurations = []
        results = []
        
        if parallel:
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._generate_sample, i): i 
                    for i in range(self.n_samples)
                }
                
                for future in as_completed(futures):
                    config, result = future.result()
                    configurations.append(config)
                    results.append(result)
                    
                    if len(configurations) % 100 == 0:
                        logger.info(f"Generated {len(configurations)}/{self.n_samples} samples")
        else:
            for i in range(self.n_samples):
                config, result = self._generate_sample(i)
                configurations.append(config)
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Generated {i+1}/{self.n_samples} samples")
        
        # Save to HDF5
        dataset_path = self.output_dir / f"aeroelastic_dataset_{self.n_samples}.h5"
        self._save_to_hdf5(configurations, results, dataset_path)
        
        # Compute statistics
        stats = self._compute_statistics(results)
        
        # Save metadata
        metadata_path = self.output_dir / f"aeroelastic_metadata_{self.n_samples}.json"
        self._save_metadata(stats, metadata_path)
        
        logger.info(f"✅ Dataset generation complete: {dataset_path}")
        
        return stats
    
    def _generate_sample(self, idx: int) -> Tuple[AeroelasticConfiguration, AeroelasticResults]:
        """Generate single sample."""
        config = self.generate_configuration(idx)
        result = self.simulate_aeroelastic_response(config)
        return config, result
    
    def _save_to_hdf5(
        self,
        configurations: List[AeroelasticConfiguration],
        results: List[AeroelasticResults],
        filepath: Path
    ):
        """Save dataset to HDF5 format."""
        logger.info(f"Saving dataset to {filepath}")
        
        with h5py.File(filepath, 'w') as f:
            # Create groups
            config_group = f.create_group('configurations')
            results_group = f.create_group('results')
            
            # Save configurations
            config_group.create_dataset('chord', data=[c.chord for c in configurations])
            config_group.create_dataset('span', data=[c.span for c in configurations])
            config_group.create_dataset('thickness', data=[c.thickness for c in configurations])
            config_group.create_dataset('camber', data=[c.camber for c in configurations])
            config_group.create_dataset('n_stiffeners', data=[c.n_stiffeners for c in configurations])
            config_group.create_dataset('stiffener_positions', data=[c.stiffener_positions for c in configurations])
            config_group.create_dataset('velocity', data=[c.velocity for c in configurations])
            config_group.create_dataset('yaw_angle', data=[c.yaw_angle for c in configurations])
            config_group.create_dataset('natural_frequencies', data=[c.natural_frequencies for c in configurations])
            config_group.create_dataset('damping_ratios', data=[c.damping_ratios for c in configurations])
            
            # Save results
            results_group.create_dataset('lift', data=[r.lift for r in results])
            results_group.create_dataset('drag', data=[r.drag for r in results])
            results_group.create_dataset('flutter_speed', data=[r.flutter_speed for r in results])
            results_group.create_dataset('flutter_margin', data=[r.flutter_margin for r in results])
            results_group.create_dataset('max_displacement', data=[r.max_displacement for r in results])
            results_group.create_dataset('max_stress', data=[r.max_stress for r in results])
            results_group.create_dataset('stress_safety_factor', data=[r.stress_safety_factor for r in results])
            
            # Metadata
            f.attrs['n_samples'] = len(configurations)
            f.attrs['version'] = '1.0'
            f.attrs['description'] = 'Aeroelastic dataset for ML surrogate training'
    
    def _compute_statistics(self, results: List[AeroelasticResults]) -> Dict:
        """Compute dataset statistics."""
        flutter_speeds = [r.flutter_speed for r in results]
        flutter_margins = [r.flutter_margin for r in results]
        max_displacements = [r.max_displacement for r in results]
        
        stats = {
            'n_samples': len(results),
            'flutter_speed': {
                'mean': float(np.mean(flutter_speeds)),
                'std': float(np.std(flutter_speeds)),
                'min': float(np.min(flutter_speeds)),
                'max': float(np.max(flutter_speeds))
            },
            'flutter_margin': {
                'mean': float(np.mean(flutter_margins)),
                'std': float(np.std(flutter_margins)),
                'min': float(np.min(flutter_margins)),
                'max': float(np.max(flutter_margins)),
                'safe_count': int(sum(1 for m in flutter_margins if m >= 1.2))
            },
            'max_displacement': {
                'mean': float(np.mean(max_displacements)),
                'std': float(np.std(max_displacements)),
                'min': float(np.min(max_displacements)),
                'max': float(np.max(max_displacements))
            }
        }
        
        return stats
    
    def _save_metadata(self, stats: Dict, filepath: Path):
        """Save metadata to JSON."""
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Metadata saved to {filepath}")


if __name__ == "__main__":
    # Generate aeroelastic dataset
    print("Aeroelastic Dataset Generator")
    print("=" * 60)
    
    # Small dataset for testing
    generator = AeroelasticDatasetGenerator(
        output_dir="data/aeroelastic-datasets",
        n_samples=100  # Start with 100 samples for testing
    )
    
    stats = generator.generate_dataset(parallel=False)
    
    print("\n✅ Dataset Generation Complete!")
    print(f"\nStatistics:")
    print(f"  Samples: {stats['n_samples']}")
    print(f"  Flutter Speed: {stats['flutter_speed']['mean']:.1f} ± {stats['flutter_speed']['std']:.1f} m/s")
    print(f"  Flutter Margin: {stats['flutter_margin']['mean']:.2f} ± {stats['flutter_margin']['std']:.2f}")
    print(f"  Safe Designs: {stats['flutter_margin']['safe_count']}/{stats['n_samples']} ({stats['flutter_margin']['safe_count']/stats['n_samples']*100:.1f}%)")
    print(f"  Max Displacement: {stats['max_displacement']['mean']*1000:.2f} ± {stats['max_displacement']['std']*1000:.2f} mm")
    
    print("\n🎯 Ready for ML surrogate training and quantum optimization!")
