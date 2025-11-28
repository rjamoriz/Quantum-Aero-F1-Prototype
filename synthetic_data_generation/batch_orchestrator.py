"""
Batch orchestration for parallel dataset generation.
Uses Dask for distributed computing and HDF5 for storage.
"""

import numpy as np
import h5py
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

from schema import AeroSample, GeometryParameters, FlowConditions
from tier1_vlm_solver import run_vlm_simulation
from sampling_strategy import ParameterSampler


class DatasetStorage:
    """Handle storage of aerodynamic dataset in HDF5 format."""
    
    def __init__(self, output_dir: str = "./synthetic_dataset"):
        """Initialize storage."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.output_dir / "metadata.json"
        self.hdf5_file = self.output_dir / "field_data.h5"
        self.scalars_file = self.output_dir / "scalars.json"
    
    def save_sample(self, sample: AeroSample, hdf5_group: Optional[h5py.Group] = None):
        """
        Save a single sample.
        Scalars go to JSON, fields go to HDF5.
        """
        # Save scalar data to JSON
        scalar_data = {
            'sample_id': sample.sample_id,
            'timestamp': sample.timestamp,
            'fidelity_tier': sample.fidelity_tier.value,
            'geometry_params': sample.geometry_params.to_dict(),
            'flow_conditions': sample.flow_conditions.to_dict(),
            'global_outputs': sample.global_outputs.to_dict(),
            'provenance': sample.provenance.to_dict(),
        }
        
        # Append to scalars file
        if self.scalars_file.exists():
            with open(self.scalars_file, 'r') as f:
                all_scalars = json.load(f)
        else:
            all_scalars = []
        
        all_scalars.append(scalar_data)
        
        with open(self.scalars_file, 'w') as f:
            json.dump(all_scalars, f, indent=2)
        
        # Save field data to HDF5
        if sample.field_outputs is not None and hdf5_group is not None:
            sample_group = hdf5_group.create_group(sample.sample_id)
            
            if sample.field_outputs.Cp is not None:
                sample_group.create_dataset('Cp', data=sample.field_outputs.Cp)
            
            if sample.field_outputs.Gamma is not None:
                sample_group.create_dataset('Gamma', data=sample.field_outputs.Gamma)
            
            if sample.field_outputs.panel_centers is not None:
                sample_group.create_dataset('panel_centers', data=sample.field_outputs.panel_centers)
            
            if sample.field_outputs.panel_normals is not None:
                sample_group.create_dataset('panel_normals', data=sample.field_outputs.panel_normals)
            
            if sample.field_outputs.panel_areas is not None:
                sample_group.create_dataset('panel_areas', data=sample.field_outputs.panel_areas)
    
    def save_batch(self, samples: List[AeroSample]):
        """Save a batch of samples."""
        with h5py.File(self.hdf5_file, 'a') as hf:
            if 'samples' not in hf:
                samples_group = hf.create_group('samples')
            else:
                samples_group = hf['samples']
            
            for sample in samples:
                self.save_sample(sample, samples_group)
    
    def load_scalars(self) -> List[Dict]:
        """Load all scalar data."""
        if not self.scalars_file.exists():
            return []
        
        with open(self.scalars_file, 'r') as f:
            return json.load(f)
    
    def load_field_data(self, sample_id: str) -> Dict[str, np.ndarray]:
        """Load field data for a specific sample."""
        with h5py.File(self.hdf5_file, 'r') as hf:
            sample_group = hf['samples'][sample_id]
            
            field_data = {}
            for key in sample_group.keys():
                field_data[key] = sample_group[key][:]
            
            return field_data
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        scalars = self.load_scalars()
        
        if not scalars:
            return {'n_samples': 0}
        
        # Extract global outputs
        CL_values = [s['global_outputs']['CL'] for s in scalars]
        CD_values = [s['global_outputs']['CD_total'] for s in scalars]
        LD_values = [s['global_outputs']['L_over_D'] for s in scalars]
        
        stats = {
            'n_samples': len(scalars),
            'CL': {
                'mean': np.mean(CL_values),
                'std': np.std(CL_values),
                'min': np.min(CL_values),
                'max': np.max(CL_values),
            },
            'CD': {
                'mean': np.mean(CD_values),
                'std': np.std(CD_values),
                'min': np.min(CD_values),
                'max': np.max(CD_values),
            },
            'L_over_D': {
                'mean': np.mean(LD_values),
                'std': np.std(LD_values),
                'min': np.min(LD_values),
                'max': np.max(LD_values),
            },
        }
        
        return stats


class BatchOrchestrator:
    """Orchestrate parallel batch generation."""
    
    def __init__(
        self,
        output_dir: str = "./synthetic_dataset",
        n_workers: int = 4
    ):
        """
        Initialize orchestrator.
        
        Args:
            output_dir: Output directory for dataset
            n_workers: Number of parallel workers
        """
        self.storage = DatasetStorage(output_dir)
        self.n_workers = n_workers
    
    def generate_tier1_batch(
        self,
        n_samples: int = 100,
        method: str = 'lhs'
    ) -> List[AeroSample]:
        """
        Generate a batch of Tier 1 (VLM) samples in parallel.
        
        Args:
            n_samples: Number of samples to generate
            method: Sampling method ('lhs' or 'stratified')
        
        Returns:
            List of AeroSample objects
        """
        print(f"=== Generating {n_samples} Tier 1 samples using {self.n_workers} workers ===\n")
        
        # Generate parameter samples
        sampler = ParameterSampler(seed=42)
        geom_samples = sampler.generate_geometry_samples(n_samples, method=method)
        flow_samples = sampler.generate_flow_conditions_samples(n_samples)
        
        # Prepare arguments for parallel execution
        args_list = [
            (geom_samples[i], flow_samples[i], i)
            for i in range(n_samples)
        ]
        
        # Run in parallel
        samples = []
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(run_vlm_simulation, *args): i 
                for i, args in enumerate(args_list)
            }
            
            # Collect results with progress bar
            with tqdm(total=n_samples, desc="Generating samples") as pbar:
                for future in as_completed(futures):
                    try:
                        sample = future.result()
                        samples.append(sample)
                        pbar.update(1)
                    except Exception as e:
                        idx = futures[future]
                        print(f"\nError in sample {idx}: {e}")
                        pbar.update(1)
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Generated {len(samples)} samples in {elapsed:.1f}s")
        print(f"  Average time per sample: {elapsed/len(samples):.2f}s")
        
        # Save batch
        print("Saving batch...")
        self.storage.save_batch(samples)
        print("✓ Batch saved")
        
        return samples
    
    def generate_dataset(
        self,
        tier1_samples: int = 1000,
        tier2_samples: int = 0,
        tier3_samples: int = 0
    ):
        """
        Generate complete multi-tier dataset.
        
        Args:
            tier1_samples: Number of Tier 1 (VLM) samples
            tier2_samples: Number of Tier 2 (transient) samples
            tier3_samples: Number of Tier 3 (CFD) samples
        """
        print("=" * 60)
        print("SYNTHETIC AERODYNAMIC DATASET GENERATION")
        print("=" * 60)
        
        total_samples = tier1_samples + tier2_samples + tier3_samples
        print(f"\nTotal samples to generate: {total_samples}")
        print(f"  Tier 1 (VLM):       {tier1_samples}")
        print(f"  Tier 2 (Transient): {tier2_samples}")
        print(f"  Tier 3 (CFD):       {tier3_samples}")
        print(f"\nUsing {self.n_workers} parallel workers")
        print()
        
        # Generate Tier 1
        if tier1_samples > 0:
            self.generate_tier1_batch(tier1_samples, method='lhs')
        
        # TODO: Implement Tier 2 and Tier 3
        if tier2_samples > 0:
            print("\n⚠ Tier 2 generation not yet implemented")
        
        if tier3_samples > 0:
            print("\n⚠ Tier 3 generation not yet implemented")
        
        # Print statistics
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        stats = self.storage.get_statistics()
        print(f"\nTotal samples: {stats['n_samples']}")
        
        if stats['n_samples'] > 0:
            print(f"\nCL statistics:")
            print(f"  Mean: {stats['CL']['mean']:.3f} ± {stats['CL']['std']:.3f}")
            print(f"  Range: [{stats['CL']['min']:.3f}, {stats['CL']['max']:.3f}]")
            
            print(f"\nCD statistics:")
            print(f"  Mean: {stats['CD']['mean']:.3f} ± {stats['CD']['std']:.3f}")
            print(f"  Range: [{stats['CD']['min']:.3f}, {stats['CD']['max']:.3f}]")
            
            print(f"\nL/D statistics:")
            print(f"  Mean: {stats['L_over_D']['mean']:.2f} ± {stats['L_over_D']['std']:.2f}")
            print(f"  Range: [{stats['L_over_D']['min']:.2f}, {stats['L_over_D']['max']:.2f}]")
        
        print(f"\nDataset saved to: {self.storage.output_dir}")
        print("=" * 60)


def main():
    """Main entry point for batch generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic aerodynamic dataset")
    parser.add_argument('--tier1', type=int, default=100, help='Number of Tier 1 samples')
    parser.add_argument('--tier2', type=int, default=0, help='Number of Tier 2 samples')
    parser.add_argument('--tier3', type=int, default=0, help='Number of Tier 3 samples')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--output', type=str, default='./synthetic_dataset', help='Output directory')
    
    args = parser.parse_args()
    
    orchestrator = BatchOrchestrator(
        output_dir=args.output,
        n_workers=args.workers
    )
    
    orchestrator.generate_dataset(
        tier1_samples=args.tier1,
        tier2_samples=args.tier2,
        tier3_samples=args.tier3
    )


if __name__ == "__main__":
    # Test with small batch
    print("=== Testing Batch Orchestrator ===\n")
    
    orchestrator = BatchOrchestrator(
        output_dir="./test_dataset",
        n_workers=2
    )
    
    # Generate small test batch
    samples = orchestrator.generate_tier1_batch(n_samples=10)
    
    print(f"\n✓ Test complete. Generated {len(samples)} samples.")
