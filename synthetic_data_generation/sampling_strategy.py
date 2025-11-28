"""
Sampling strategies for synthetic dataset generation.
Implements Latin Hypercube Sampling (LHS) and stratified sampling.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import qmc
import json

from schema import GeometryParameters, FlowConditions


class ParameterSampler:
    """Generate parameter samples using various strategies."""
    
    def __init__(self, seed: int = 42):
        """Initialize sampler with random seed."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def latin_hypercube_sampling(
        self,
        n_samples: int,
        bounds: Dict[str, Tuple[float, float]]
    ) -> np.ndarray:
        """
        Generate samples using Latin Hypercube Sampling.
        
        Args:
            n_samples: Number of samples to generate
            bounds: Dictionary of parameter bounds {param_name: (low, high)}
        
        Returns:
            samples: (n_samples, n_params) array
        """
        n_params = len(bounds)
        param_names = list(bounds.keys())
        
        # Create LHS sampler
        sampler = qmc.LatinHypercube(d=n_params, seed=self.seed)
        
        # Generate samples in [0, 1]^n_params
        unit_samples = sampler.random(n=n_samples)
        
        # Scale to actual bounds
        samples = np.zeros_like(unit_samples)
        for i, param_name in enumerate(param_names):
            low, high = bounds[param_name]
            samples[:, i] = low + unit_samples[:, i] * (high - low)
        
        return samples, param_names
    
    def stratified_sampling(
        self,
        n_samples: int,
        bounds: Dict[str, Tuple[float, float]],
        discrete_params: List[str] = None
    ) -> np.ndarray:
        """
        Generate stratified samples with special handling for discrete parameters.
        
        Args:
            n_samples: Number of samples
            bounds: Parameter bounds
            discrete_params: List of parameter names that are discrete/binary
        
        Returns:
            samples: (n_samples, n_params) array
        """
        if discrete_params is None:
            discrete_params = []
        
        param_names = list(bounds.keys())
        n_params = len(param_names)
        samples = np.zeros((n_samples, n_params))
        
        for i, param_name in enumerate(param_names):
            low, high = bounds[param_name]
            
            if param_name in discrete_params:
                # Binary/discrete: stratify into bins
                if high - low <= 1.0:  # Binary
                    # Alternate between 0 and 1
                    samples[:, i] = np.array([0.0, 1.0] * (n_samples // 2 + 1))[:n_samples]
                else:
                    # Discrete values
                    n_values = int(high - low) + 1
                    values = np.linspace(low, high, n_values)
                    samples[:, i] = self.rng.choice(values, size=n_samples)
            else:
                # Continuous: stratified uniform
                strata = np.linspace(low, high, n_samples + 1)
                for j in range(n_samples):
                    samples[j, i] = self.rng.uniform(strata[j], strata[j+1])
        
        return samples, param_names
    
    def generate_geometry_samples(
        self,
        n_samples: int,
        method: str = "lhs"
    ) -> List[GeometryParameters]:
        """
        Generate geometry parameter samples.
        
        Args:
            n_samples: Number of samples
            method: "lhs" or "stratified"
        
        Returns:
            List of GeometryParameters
        """
        bounds = GeometryParameters.get_bounds()
        discrete_params = ['DRS_open']
        
        if method == "lhs":
            samples, param_names = self.latin_hypercube_sampling(n_samples, bounds)
        elif method == "stratified":
            samples, param_names = self.stratified_sampling(
                n_samples, bounds, discrete_params
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert to GeometryParameters objects
        geom_samples = []
        for i in range(n_samples):
            params_dict = {
                name: float(samples[i, j]) 
                for j, name in enumerate(param_names)
            }
            # Convert DRS_open to boolean
            params_dict['DRS_open'] = bool(params_dict['DRS_open'] > 0.5)
            
            geom_samples.append(GeometryParameters(**params_dict))
        
        return geom_samples
    
    def generate_flow_conditions_samples(
        self,
        n_samples: int,
        V_inf_range: Tuple[float, float] = (50.0, 90.0),
        yaw_range: Tuple[float, float] = (-5.0, 5.0),
        ground_gap_range: Tuple[float, float] = (10.0, 50.0)
    ) -> List[FlowConditions]:
        """
        Generate flow condition samples.
        
        Args:
            n_samples: Number of samples
            V_inf_range: Velocity range (m/s)
            yaw_range: Yaw angle range (deg)
            ground_gap_range: Ground clearance range (mm)
        
        Returns:
            List of FlowConditions
        """
        # LHS for flow conditions
        bounds = {
            'V_inf': V_inf_range,
            'yaw': yaw_range,
            'ground_gap': ground_gap_range
        }
        
        samples, param_names = self.latin_hypercube_sampling(n_samples, bounds)
        
        # Convert to FlowConditions objects
        flow_samples = []
        for i in range(n_samples):
            V_inf = samples[i, 0]
            yaw = samples[i, 1]
            ground_gap = samples[i, 2]
            
            flow = FlowConditions.standard_conditions(V_inf=V_inf, ground_gap=ground_gap)
            flow.yaw = yaw
            
            flow_samples.append(flow)
        
        return flow_samples


class DatasetPlan:
    """Plan for multi-tier dataset generation."""
    
    def __init__(self):
        """Initialize dataset plan."""
        self.tiers = {
            'tier1_fast': {
                'n_samples': 5000,
                'method': 'lhs',
                'description': 'Fast VLM samples for bulk data'
            },
            'tier2_transient': {
                'n_samples': 1000,
                'method': 'stratified',
                'description': 'Transient scenarios (DRS, gusts)'
            },
            'tier3_high_fidelity': {
                'n_samples': 100,
                'method': 'stratified',
                'description': 'High-fidelity CFD validation'
            }
        }
    
    def generate_tier1_plan(self, n_samples: int = 5000) -> Dict:
        """Generate plan for Tier 1 (fast VLM)."""
        sampler = ParameterSampler(seed=42)
        
        geom_samples = sampler.generate_geometry_samples(n_samples, method='lhs')
        flow_samples = sampler.generate_flow_conditions_samples(n_samples)
        
        return {
            'tier': 1,
            'n_samples': n_samples,
            'geometry_samples': [g.to_dict() for g in geom_samples],
            'flow_samples': [f.to_dict() for f in flow_samples],
            'method': 'lhs',
            'estimated_time_per_sample': 5.0,  # seconds
            'estimated_total_time': n_samples * 5.0 / 3600.0,  # hours
        }
    
    def generate_tier2_plan(self, n_samples: int = 1000) -> Dict:
        """Generate plan for Tier 2 (transient)."""
        sampler = ParameterSampler(seed=43)
        
        # Focus on transient scenarios
        geom_samples = sampler.generate_geometry_samples(n_samples, method='stratified')
        
        # Ensure mix of DRS states
        n_drs_open = n_samples // 2
        for i in range(n_drs_open):
            geom_samples[i].DRS_open = True
        
        flow_samples = sampler.generate_flow_conditions_samples(n_samples)
        
        return {
            'tier': 2,
            'n_samples': n_samples,
            'geometry_samples': [g.to_dict() for g in geom_samples],
            'flow_samples': [f.to_dict() for f in flow_samples],
            'method': 'stratified',
            'transient_scenarios': ['DRS_activation', 'gust_response'],
            'estimated_time_per_sample': 30.0,  # seconds
            'estimated_total_time': n_samples * 30.0 / 3600.0,  # hours
        }
    
    def generate_tier3_plan(self, n_samples: int = 100) -> Dict:
        """Generate plan for Tier 3 (high-fidelity CFD)."""
        sampler = ParameterSampler(seed=44)
        
        # Select representative samples across parameter space
        geom_samples = sampler.generate_geometry_samples(n_samples, method='stratified')
        flow_samples = sampler.generate_flow_conditions_samples(n_samples)
        
        return {
            'tier': 3,
            'n_samples': n_samples,
            'geometry_samples': [g.to_dict() for g in geom_samples],
            'flow_samples': [f.to_dict() for f in flow_samples],
            'method': 'stratified',
            'solvers': ['OpenFOAM_RANS', 'SU2_RANS'],
            'estimated_time_per_sample': 3600.0,  # 1 hour per sample
            'estimated_total_time': n_samples * 3600.0 / 3600.0,  # hours
        }
    
    def save_plan(self, filename: str = "dataset_generation_plan.json"):
        """Save complete dataset generation plan."""
        plan = {
            'tier1': self.generate_tier1_plan(),
            'tier2': self.generate_tier2_plan(),
            'tier3': self.generate_tier3_plan(),
            'total_samples': 5000 + 1000 + 100,
            'total_estimated_time_hours': (
                5000 * 5.0 / 3600.0 +
                1000 * 30.0 / 3600.0 +
                100 * 3600.0 / 3600.0
            )
        }
        
        with open(filename, 'w') as f:
            json.dump(plan, f, indent=2)
        
        print(f"Dataset generation plan saved to {filename}")
        print(f"Total samples: {plan['total_samples']}")
        print(f"Estimated time: {plan['total_estimated_time_hours']:.1f} hours")
        
        return plan


def test_sampling():
    """Test sampling strategies."""
    print("=== Testing Sampling Strategies ===\n")
    
    sampler = ParameterSampler(seed=42)
    
    # Generate geometry samples
    print("Generating 10 geometry samples using LHS...")
    geom_samples = sampler.generate_geometry_samples(10, method='lhs')
    
    print(f"Generated {len(geom_samples)} samples")
    print("\nFirst sample:")
    print(json.dumps(geom_samples[0].to_dict(), indent=2))
    
    # Generate flow samples
    print("\n\nGenerating 10 flow condition samples...")
    flow_samples = sampler.generate_flow_conditions_samples(10)
    
    print(f"Generated {len(flow_samples)} samples")
    print("\nFirst sample:")
    print(json.dumps(flow_samples[0].to_dict(), indent=2))
    
    # Generate complete plan
    print("\n\n=== Generating Complete Dataset Plan ===")
    planner = DatasetPlan()
    plan = planner.save_plan()
    
    return geom_samples, flow_samples, plan


if __name__ == "__main__":
    geom_samples, flow_samples, plan = test_sampling()
