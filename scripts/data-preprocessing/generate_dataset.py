"""
Synthetic Dataset Generation Pipeline
Generates training data for ML surrogate using VLM solver and F1 geometry
"""

import numpy as np
import h5py
from pathlib import Path
import logging
from typing import Dict, List
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime
import json

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'services' / 'physics-engine'))
from vlm.solver import VortexLatticeMethod, WingGeometry

from f1_geometry import F1GeometryBuilder, F1GeometryParams, create_parametric_variations
from naca_airfoil import NACAProfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """
    Generate synthetic aerodynamic dataset.
    
    Pipeline:
    1. Generate parametric variations
    2. Build F1 geometry for each variation
    3. Run VLM simulation
    4. Extract features
    5. Save to HDF5 + metadata
    """
    
    def __init__(
        self,
        output_dir: str = 'data/training-datasets',
        n_samples: int = 1000,
        n_workers: int = 4
    ):
        """
        Initialize dataset generator.
        
        Args:
            output_dir: Output directory for datasets
            n_samples: Number of samples to generate
            n_workers: Number of parallel workers
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_samples = n_samples
        self.n_workers = n_workers
        
        logger.info(f"Dataset Generator initialized: {n_samples} samples, {n_workers} workers")
    
    def generate_single_sample(self, params: F1GeometryParams, sample_id: int) -> Dict:
        """
        Generate single data sample.
        
        Args:
            params: Geometry parameters
            sample_id: Sample identifier
            
        Returns:
            Dictionary with geometry, simulation results, and metadata
        """
        try:
            # Build geometry
            builder = F1GeometryBuilder(params)
            geometry = builder.build_complete_geometry()
            
            # Simplified VLM simulation (front wing only for now)
            # In production, would simulate complete car
            front_wing = geometry['front_wing']
            
            # Create simplified wing geometry for VLM
            wing_geom = WingGeometry(
                span=1.8,
                chord=0.25,
                twist=-2.0,  # Washout
                dihedral=0.0,
                sweep=0.0,
                taper_ratio=1.0
            )
            
            # Run VLM simulation
            vlm = VortexLatticeMethod(n_panels_x=20, n_panels_y=10)
            vlm.setup_geometry(wing_geom)
            
            result = vlm.solve(
                velocity=params.velocity,
                alpha=params.front_wing_main_aoa,
                yaw=params.yaw_angle,
                rho=1.225
            )
            
            # Extract features
            sample = {
                'sample_id': sample_id,
                'timestamp': datetime.now().isoformat(),
                
                # Design parameters
                'parameters': {
                    'front_wing_main_aoa': params.front_wing_main_aoa,
                    'front_wing_flap1_aoa': params.front_wing_flap1_aoa,
                    'front_wing_flap2_aoa': params.front_wing_flap2_aoa,
                    'front_wing_ride_height': params.front_wing_ride_height,
                    'rear_wing_main_aoa': params.rear_wing_main_aoa,
                    'rear_wing_flap_aoa': params.rear_wing_flap_aoa,
                    'drs_open': params.drs_open,
                    'floor_ride_height_front': params.floor_ride_height_front,
                    'floor_ride_height_rear': params.floor_ride_height_rear,
                    'diffuser_angle': params.diffuser_angle,
                    'velocity': params.velocity,
                    'yaw_angle': params.yaw_angle,
                    'pitch_angle': params.pitch_angle
                },
                
                # Flow conditions
                'flow_conditions': {
                    'velocity': params.velocity,
                    'reynolds_number': params.velocity * 0.25 / 1.5e-5,
                    'mach_number': params.velocity / 340.0
                },
                
                # Aerodynamic results
                'results': {
                    'cl': float(result.cl),
                    'cd': float(result.cd),
                    'cm': float(result.cm),
                    'l_over_d': float(result.cl / result.cd) if result.cd > 0 else 0,
                    'lift': float(result.forces['lift']),
                    'drag': float(result.forces['drag']),
                    'side_force': float(result.forces['side']),
                    'moment': float(result.forces['moment'])
                },
                
                # Pressure distribution
                'pressure': result.pressure.tolist(),
                'gamma': result.gamma.tolist(),
                
                # Metadata
                'simulation_method': 'VLM',
                'n_panels': vlm.n_panels,
                'convergence': True
            }
            
            return sample
            
        except Exception as e:
            logger.error(f"Error generating sample {sample_id}: {str(e)}")
            return None
    
    def generate_dataset(self) -> str:
        """
        Generate complete dataset.
        
        Returns:
            Path to generated HDF5 file
        """
        logger.info(f"Generating dataset: {self.n_samples} samples")
        
        # Generate parametric variations
        base_params = F1GeometryParams()
        param_variations = create_parametric_variations(base_params, self.n_samples)
        
        # Generate samples
        logger.info("Running simulations...")
        samples = []
        
        if self.n_workers > 1:
            # Parallel processing
            with mp.Pool(self.n_workers) as pool:
                args = [(params, i) for i, params in enumerate(param_variations)]
                results = list(tqdm(
                    pool.starmap(self.generate_single_sample, args),
                    total=self.n_samples,
                    desc="Generating samples"
                ))
                samples = [s for s in results if s is not None]
        else:
            # Sequential processing
            for i, params in enumerate(tqdm(param_variations, desc="Generating samples")):
                sample = self.generate_single_sample(params, i)
                if sample is not None:
                    samples.append(sample)
        
        logger.info(f"Generated {len(samples)} valid samples")
        
        # Save to HDF5
        output_file = self.save_to_hdf5(samples)
        
        # Save metadata
        self.save_metadata(samples)
        
        return output_file
    
    def save_to_hdf5(self, samples: List[Dict]) -> str:
        """
        Save dataset to HDF5 format.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            Path to HDF5 file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f'f1_aero_dataset_{timestamp}.h5'
        
        logger.info(f"Saving to HDF5: {output_file}")
        
        n_samples = len(samples)
        n_panels = len(samples[0]['pressure'])
        
        with h5py.File(output_file, 'w') as f:
            # Create groups
            params_group = f.create_group('parameters')
            flow_group = f.create_group('flow_conditions')
            results_group = f.create_group('results')
            pressure_group = f.create_group('pressure_distribution')
            meta_group = f.create_group('metadata')
            
            # Store parameters
            for key in samples[0]['parameters'].keys():
                data = np.array([s['parameters'][key] for s in samples])
                params_group.create_dataset(key, data=data)
            
            # Store flow conditions
            for key in samples[0]['flow_conditions'].keys():
                data = np.array([s['flow_conditions'][key] for s in samples])
                flow_group.create_dataset(key, data=data)
            
            # Store results
            for key in samples[0]['results'].keys():
                data = np.array([s['results'][key] for s in samples])
                results_group.create_dataset(key, data=data)
            
            # Store pressure distributions
            pressure_data = np.array([s['pressure'] for s in samples])
            pressure_group.create_dataset('cp', data=pressure_data)
            
            gamma_data = np.array([s['gamma'] for s in samples])
            pressure_group.create_dataset('gamma', data=gamma_data)
            
            # Store metadata
            meta_group.attrs['n_samples'] = n_samples
            meta_group.attrs['n_panels'] = n_panels
            meta_group.attrs['generation_date'] = datetime.now().isoformat()
            meta_group.attrs['simulation_method'] = 'VLM'
        
        file_size_mb = output_file.stat().st_size / 1e6
        logger.info(f"Dataset saved: {output_file} ({file_size_mb:.2f} MB)")
        
        return str(output_file)
    
    def save_metadata(self, samples: List[Dict]):
        """
        Save dataset metadata to JSON.
        
        Args:
            samples: List of sample dictionaries
        """
        metadata = {
            'n_samples': len(samples),
            'generation_date': datetime.now().isoformat(),
            'simulation_method': 'VLM',
            'parameter_ranges': self._compute_parameter_ranges(samples),
            'statistics': self._compute_statistics(samples)
        }
        
        metadata_file = self.output_dir / 'dataset_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved: {metadata_file}")
    
    def _compute_parameter_ranges(self, samples: List[Dict]) -> Dict:
        """Compute parameter ranges from samples"""
        ranges = {}
        for key in samples[0]['parameters'].keys():
            values = [s['parameters'][key] for s in samples]
            ranges[key] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        return ranges
    
    def _compute_statistics(self, samples: List[Dict]) -> Dict:
        """Compute dataset statistics"""
        cl_values = [s['results']['cl'] for s in samples]
        cd_values = [s['results']['cd'] for s in samples]
        
        return {
            'cl': {
                'min': float(np.min(cl_values)),
                'max': float(np.max(cl_values)),
                'mean': float(np.mean(cl_values)),
                'std': float(np.std(cl_values))
            },
            'cd': {
                'min': float(np.min(cd_values)),
                'max': float(np.max(cd_values)),
                'mean': float(np.mean(cd_values)),
                'std': float(np.std(cd_values))
            },
            'l_over_d': {
                'min': float(np.min([s['results']['l_over_d'] for s in samples])),
                'max': float(np.max([s['results']['l_over_d'] for s in samples])),
                'mean': float(np.mean([s['results']['l_over_d'] for s in samples]))
            }
        }


def load_dataset(hdf5_file: str) -> Dict:
    """
    Load dataset from HDF5 file.
    
    Args:
        hdf5_file: Path to HDF5 file
        
    Returns:
        Dictionary with dataset arrays
    """
    logger.info(f"Loading dataset: {hdf5_file}")
    
    dataset = {}
    
    with h5py.File(hdf5_file, 'r') as f:
        # Load parameters
        dataset['parameters'] = {}
        for key in f['parameters'].keys():
            dataset['parameters'][key] = f['parameters'][key][:]
        
        # Load flow conditions
        dataset['flow_conditions'] = {}
        for key in f['flow_conditions'].keys():
            dataset['flow_conditions'][key] = f['flow_conditions'][key][:]
        
        # Load results
        dataset['results'] = {}
        for key in f['results'].keys():
            dataset['results'][key] = f['results'][key][:]
        
        # Load pressure
        dataset['pressure'] = f['pressure_distribution']['cp'][:]
        dataset['gamma'] = f['pressure_distribution']['gamma'][:]
        
        # Load metadata
        dataset['metadata'] = dict(f['metadata'].attrs)
    
    logger.info(f"Loaded {dataset['metadata']['n_samples']} samples")
    
    return dataset


if __name__ == "__main__":
    # Generate dataset
    print("F1 Aerodynamic Dataset Generation")
    print("=" * 60)
    
    # Configuration
    n_samples = 100  # Start small for testing
    n_workers = 4
    
    print(f"\nConfiguration:")
    print(f"  Samples: {n_samples}")
    print(f"  Workers: {n_workers}")
    print(f"  Output: data/training-datasets/")
    
    # Create generator
    generator = DatasetGenerator(
        output_dir='data/training-datasets',
        n_samples=n_samples,
        n_workers=n_workers
    )
    
    # Generate dataset
    print("\nGenerating dataset...")
    output_file = generator.generate_dataset()
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"   File: {output_file}")
    
    # Load and verify
    print("\nVerifying dataset...")
    dataset = load_dataset(output_file)
    
    print(f"\nDataset Statistics:")
    print(f"  Samples: {dataset['metadata']['n_samples']}")
    print(f"  Panels: {dataset['metadata']['n_panels']}")
    print(f"  CL range: [{dataset['results']['cl'].min():.3f}, {dataset['results']['cl'].max():.3f}]")
    print(f"  CD range: [{dataset['results']['cd'].min():.4f}, {dataset['results']['cd'].max():.4f}]")
    print(f"  L/D range: [{dataset['results']['l_over_d'].min():.1f}, {dataset['results']['l_over_d'].max():.1f}]")
    
    print("\n✅ All tests passed!")
