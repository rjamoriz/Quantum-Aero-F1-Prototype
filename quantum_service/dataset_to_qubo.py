"""
Dataset to QUBO Converter
Transforms synthetic aerodynamic datasets into QUBO/QAOA formulations
for quantum optimization
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add synthetic_data_generation to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'synthetic_data_generation'))

from schema import GeometryParameters, FlowConditions


@dataclass
class QUBOFormulation:
    """QUBO problem formulation"""
    Q_matrix: np.ndarray
    offset: float
    variable_mapping: Dict[str, List[int]]  # Maps parameter names to qubit indices
    num_qubits: int
    objective_type: str
    constraints: List[Dict]


class DatasetToQUBOConverter:
    """
    Converts synthetic aerodynamic datasets to QUBO formulations
    for quantum optimization
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize converter with dataset
        
        Args:
            dataset_path: Path to scalars.json from synthetic dataset
        """
        self.dataset_path = dataset_path
        self.samples = self._load_dataset()
        self.surrogate_model = None
        
    def _load_dataset(self) -> List[Dict]:
        """Load dataset from JSON"""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def train_surrogate(self, model_type: str = 'polynomial') -> None:
        """
        Train a surrogate model on the dataset
        
        Args:
            model_type: Type of surrogate ('polynomial', 'rbf', 'gp')
        """
        # Extract features and targets
        X = []
        y_CL = []
        y_CD = []
        y_LD = []
        
        for sample in self.samples:
            geom = sample['geometry_params']
            outputs = sample['global_outputs']
            
            # Feature vector (16D geometry parameters)
            features = [
                geom['main_plane_chord'],
                geom['main_plane_span'],
                geom['main_plane_angle_deg'],
                geom['flap1_angle_deg'],
                geom['flap2_angle_deg'],
                geom['endplate_height'],
                geom['rear_wing_chord'],
                geom['rear_wing_span'],
                geom['rear_wing_angle_deg'],
                geom['beam_wing_angle'],
                geom['floor_gap'],
                geom['diffuser_angle'],
                geom['diffuser_length'],
                geom['sidepod_width'],
                geom['sidepod_undercut'],
                float(geom['DRS_open'])
            ]
            
            X.append(features)
            y_CL.append(outputs['CL'])
            y_CD.append(outputs['CD_total'])
            y_LD.append(outputs['L_over_D'])
        
        X = np.array(X)
        y_CL = np.array(y_CL)
        y_CD = np.array(y_CD)
        y_LD = np.array(y_LD)
        
        if model_type == 'polynomial':
            # Fit polynomial regression (degree 2)
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import Ridge
            
            poly = PolynomialFeatures(degree=2, include_bias=True)
            X_poly = poly.fit_transform(X)
            
            # Train separate models for each output
            model_CL = Ridge(alpha=1.0).fit(X_poly, y_CL)
            model_CD = Ridge(alpha=1.0).fit(X_poly, y_CD)
            model_LD = Ridge(alpha=1.0).fit(X_poly, y_LD)
            
            self.surrogate_model = {
                'type': 'polynomial',
                'poly_features': poly,
                'model_CL': model_CL,
                'model_CD': model_CD,
                'model_LD': model_LD,
                'feature_names': self._get_feature_names()
            }
            
            print(f"✓ Surrogate model trained on {len(X)} samples")
            print(f"  CL R²: {model_CL.score(X_poly, y_CL):.4f}")
            print(f"  CD R²: {model_CD.score(X_poly, y_CD):.4f}")
            print(f"  L/D R²: {model_LD.score(X_poly, y_LD):.4f}")
    
    def _get_feature_names(self) -> List[str]:
        """Get geometry parameter names"""
        return [
            'main_plane_chord', 'main_plane_span', 'main_plane_angle_deg',
            'flap1_angle_deg', 'flap2_angle_deg', 'endplate_height',
            'rear_wing_chord', 'rear_wing_span', 'rear_wing_angle_deg',
            'beam_wing_angle', 'floor_gap', 'diffuser_angle',
            'diffuser_length', 'sidepod_width', 'sidepod_undercut', 'DRS_open'
        ]
    
    def formulate_qubo(
        self,
        objective: str = 'maximize_L_over_D',
        design_variables: Optional[List[str]] = None,
        constraints: Optional[List[Dict]] = None,
        discretization_bits: int = 4
    ) -> QUBOFormulation:
        """
        Formulate QUBO problem from dataset
        
        Args:
            objective: Optimization objective
            design_variables: List of parameters to optimize (None = all)
            constraints: List of constraint dicts
            discretization_bits: Bits per variable for binary encoding
        
        Returns:
            QUBOFormulation object
        """
        if self.surrogate_model is None:
            print("Training surrogate model...")
            self.train_surrogate()
        
        # Default: optimize wing angles and floor gap
        if design_variables is None:
            design_variables = [
                'main_plane_angle_deg',
                'flap1_angle_deg',
                'rear_wing_angle_deg',
                'floor_gap'
            ]
        
        # Get parameter bounds from dataset
        param_bounds = self._get_parameter_bounds(design_variables)
        
        # Create binary encoding
        variable_mapping = {}
        current_qubit = 0
        
        for var in design_variables:
            qubit_indices = list(range(current_qubit, current_qubit + discretization_bits))
            variable_mapping[var] = qubit_indices
            current_qubit += discretization_bits
        
        num_qubits = current_qubit
        
        # Initialize QUBO matrix
        Q = np.zeros((num_qubits, num_qubits))
        
        # Build QUBO based on objective
        if objective == 'maximize_L_over_D':
            Q, offset = self._build_ld_maximization_qubo(
                Q, variable_mapping, param_bounds, discretization_bits
            )
        elif objective == 'minimize_drag':
            Q, offset = self._build_drag_minimization_qubo(
                Q, variable_mapping, param_bounds, discretization_bits
            )
        elif objective == 'balance_optimization':
            Q, offset = self._build_balance_optimization_qubo(
                Q, variable_mapping, param_bounds, discretization_bits
            )
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Add constraints as penalty terms
        if constraints:
            Q = self._add_constraint_penalties(Q, constraints, variable_mapping)
        
        return QUBOFormulation(
            Q_matrix=Q,
            offset=offset,
            variable_mapping=variable_mapping,
            num_qubits=num_qubits,
            objective_type=objective,
            constraints=constraints or []
        )
    
    def _get_parameter_bounds(self, variables: List[str]) -> Dict[str, Tuple[float, float]]:
        """Extract parameter bounds from dataset"""
        bounds = {}
        
        for var in variables:
            values = []
            for sample in self.samples:
                values.append(sample['geometry_params'][var])
            
            bounds[var] = (min(values), max(values))
        
        return bounds
    
    def _build_ld_maximization_qubo(
        self,
        Q: np.ndarray,
        variable_mapping: Dict,
        param_bounds: Dict,
        bits: int
    ) -> Tuple[np.ndarray, float]:
        """
        Build QUBO for L/D maximization
        Minimizes: -L/D (since QUBO minimizes)
        """
        # Sample grid of configurations
        n_samples = 1000
        X_samples = self._generate_sample_grid(variable_mapping, param_bounds, bits, n_samples)
        
        # Predict L/D for each
        poly = self.surrogate_model['poly_features']
        model_LD = self.surrogate_model['model_LD']
        
        X_poly = poly.transform(X_samples)
        LD_pred = model_LD.predict(X_poly)
        
        # Fit quadratic approximation to -L/D
        # -L/D ≈ x^T Q x + offset
        
        # Use least squares to fit Q matrix
        # This is a simplified approach - for production, use proper QUBO compilation
        
        # For now, use a heuristic: penalize deviation from best known configuration
        best_idx = np.argmax(LD_pred)
        best_config = X_samples[best_idx]
        
        # Create penalty for distance from best
        for i, (var, qubits) in enumerate(variable_mapping.items()):
            target_value = best_config[list(variable_mapping.keys()).index(var)]
            param_min, param_max = param_bounds[var]
            
            # Binary encoding target
            target_binary = self._value_to_binary(target_value, param_min, param_max, bits)
            
            # Penalize bits that differ from target
            for j, qubit in enumerate(qubits):
                if target_binary[j] == 0:
                    Q[qubit, qubit] += 1.0  # Penalize if bit is 1
                else:
                    Q[qubit, qubit] -= 1.0  # Reward if bit is 1
        
        offset = -np.max(LD_pred)  # Offset to make minimum near zero
        
        return Q, offset
    
    def _build_drag_minimization_qubo(
        self,
        Q: np.ndarray,
        variable_mapping: Dict,
        param_bounds: Dict,
        bits: int
    ) -> Tuple[np.ndarray, float]:
        """Build QUBO for drag minimization"""
        # Similar to L/D but minimize CD
        n_samples = 1000
        X_samples = self._generate_sample_grid(variable_mapping, param_bounds, bits, n_samples)
        
        poly = self.surrogate_model['poly_features']
        model_CD = self.surrogate_model['model_CD']
        
        X_poly = poly.transform(X_samples)
        CD_pred = model_CD.predict(X_poly)
        
        best_idx = np.argmin(CD_pred)
        best_config = X_samples[best_idx]
        
        for i, (var, qubits) in enumerate(variable_mapping.items()):
            target_value = best_config[list(variable_mapping.keys()).index(var)]
            param_min, param_max = param_bounds[var]
            
            target_binary = self._value_to_binary(target_value, param_min, param_max, bits)
            
            for j, qubit in enumerate(qubits):
                if target_binary[j] == 0:
                    Q[qubit, qubit] += 1.0
                else:
                    Q[qubit, qubit] -= 1.0
        
        offset = np.min(CD_pred)
        
        return Q, offset
    
    def _build_balance_optimization_qubo(
        self,
        Q: np.ndarray,
        variable_mapping: Dict,
        param_bounds: Dict,
        bits: int,
        target_balance: float = 0.40
    ) -> Tuple[np.ndarray, float]:
        """Build QUBO for balance optimization (target 40% front)"""
        n_samples = 1000
        X_samples = self._generate_sample_grid(variable_mapping, param_bounds, bits, n_samples)
        
        # Predict balance for each configuration
        # Balance = downforce_front / (downforce_front + downforce_rear)
        
        # For simplicity, find configuration closest to target balance in dataset
        best_balance_error = float('inf')
        best_config = None
        
        for sample in self.samples:
            balance = sample['global_outputs']['balance']
            error = abs(balance - target_balance)
            
            if error < best_balance_error:
                best_balance_error = error
                best_config = sample['geometry_params']
        
        # Encode best configuration
        for var, qubits in variable_mapping.items():
            target_value = best_config[var]
            param_min, param_max = param_bounds[var]
            
            target_binary = self._value_to_binary(target_value, param_min, param_max, bits)
            
            for j, qubit in enumerate(qubits):
                if target_binary[j] == 0:
                    Q[qubit, qubit] += 1.0
                else:
                    Q[qubit, qubit] -= 1.0
        
        offset = best_balance_error
        
        return Q, offset
    
    def _generate_sample_grid(
        self,
        variable_mapping: Dict,
        param_bounds: Dict,
        bits: int,
        n_samples: int
    ) -> np.ndarray:
        """Generate sample grid for QUBO construction"""
        # Create full feature vector with defaults
        default_sample = self.samples[0]['geometry_params']
        
        samples = []
        for _ in range(n_samples):
            sample = []
            for feature_name in self._get_feature_names():
                if feature_name in variable_mapping:
                    # Sample from bounds
                    param_min, param_max = param_bounds[feature_name]
                    value = np.random.uniform(param_min, param_max)
                else:
                    # Use default
                    value = default_sample[feature_name]
                
                sample.append(value)
            
            samples.append(sample)
        
        return np.array(samples)
    
    def _value_to_binary(
        self,
        value: float,
        min_val: float,
        max_val: float,
        bits: int
    ) -> List[int]:
        """Convert continuous value to binary encoding"""
        # Normalize to [0, 1]
        normalized = (value - min_val) / (max_val - min_val)
        
        # Convert to integer in [0, 2^bits - 1]
        max_int = (2 ** bits) - 1
        int_value = int(normalized * max_int)
        
        # Convert to binary
        binary = [int(b) for b in format(int_value, f'0{bits}b')]
        
        return binary
    
    def _binary_to_value(
        self,
        binary: List[int],
        min_val: float,
        max_val: float
    ) -> float:
        """Convert binary encoding to continuous value"""
        bits = len(binary)
        max_int = (2 ** bits) - 1
        
        # Binary to integer
        int_value = int(''.join(map(str, binary)), 2)
        
        # Normalize
        normalized = int_value / max_int
        
        # Scale to range
        value = min_val + normalized * (max_val - min_val)
        
        return value
    
    def _add_constraint_penalties(
        self,
        Q: np.ndarray,
        constraints: List[Dict],
        variable_mapping: Dict,
        penalty_weight: float = 10.0
    ) -> np.ndarray:
        """Add constraint penalties to QUBO matrix"""
        # Constraints are enforced as penalty terms
        # For each violated constraint, add penalty_weight to objective
        
        for constraint in constraints:
            param = constraint['parameter']
            operator = constraint['operator']
            value = constraint['value']
            weight = constraint.get('weight', 1.0) * penalty_weight
            
            if param in variable_mapping:
                # Add penalty term (simplified - proper implementation would be more complex)
                qubits = variable_mapping[param]
                for qubit in qubits:
                    Q[qubit, qubit] += weight
        
        return Q
    
    def decode_solution(
        self,
        binary_solution: List[int],
        variable_mapping: Dict,
        param_bounds: Dict
    ) -> Dict[str, float]:
        """
        Decode binary solution to geometry parameters
        
        Args:
            binary_solution: Binary string solution from quantum computer
            variable_mapping: Mapping of variables to qubit indices
            param_bounds: Parameter bounds
        
        Returns:
            Dictionary of parameter values
        """
        decoded = {}
        
        for var, qubits in variable_mapping.items():
            binary_var = [binary_solution[q] for q in qubits]
            min_val, max_val = param_bounds[var]
            
            value = self._binary_to_value(binary_var, min_val, max_val)
            decoded[var] = value
        
        return decoded
    
    def predict_performance(self, geometry_params: Dict) -> Dict[str, float]:
        """
        Predict aerodynamic performance for given geometry
        
        Args:
            geometry_params: Dictionary of geometry parameters
        
        Returns:
            Dictionary with CL, CD, L_over_D predictions
        """
        if self.surrogate_model is None:
            raise ValueError("Surrogate model not trained")
        
        # Create feature vector
        features = []
        for name in self._get_feature_names():
            features.append(geometry_params.get(name, 0.0))
        
        X = np.array([features])
        
        poly = self.surrogate_model['poly_features']
        X_poly = poly.transform(X)
        
        CL = self.surrogate_model['model_CL'].predict(X_poly)[0]
        CD = self.surrogate_model['model_CD'].predict(X_poly)[0]
        LD = self.surrogate_model['model_LD'].predict(X_poly)[0]
        
        return {
            'CL': float(CL),
            'CD': float(CD),
            'L_over_D': float(LD)
        }
    
    def export_qubo(self, formulation: QUBOFormulation, output_path: str) -> None:
        """Export QUBO formulation to JSON"""
        export_data = {
            'Q_matrix': formulation.Q_matrix.tolist(),
            'offset': float(formulation.offset),
            'variable_mapping': formulation.variable_mapping,
            'num_qubits': formulation.num_qubits,
            'objective_type': formulation.objective_type,
            'constraints': formulation.constraints
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ QUBO formulation exported to {output_path}")
        print(f"  Qubits: {formulation.num_qubits}")
        print(f"  Objective: {formulation.objective_type}")


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert synthetic dataset to QUBO formulation")
    parser.add_argument('--dataset', type=str, required=True, help='Path to scalars.json')
    parser.add_argument('--objective', type=str, default='maximize_L_over_D',
                       choices=['maximize_L_over_D', 'minimize_drag', 'balance_optimization'])
    parser.add_argument('--output', type=str, default='qubo_formulation.json')
    parser.add_argument('--bits', type=int, default=4, help='Discretization bits per variable')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DATASET TO QUBO CONVERTER")
    print("=" * 60)
    
    # Load and convert
    converter = DatasetToQUBOConverter(args.dataset)
    
    print(f"\nLoaded {len(converter.samples)} samples")
    
    # Train surrogate
    print("\nTraining surrogate model...")
    converter.train_surrogate()
    
    # Formulate QUBO
    print(f"\nFormulating QUBO for objective: {args.objective}")
    formulation = converter.formulate_qubo(
        objective=args.objective,
        discretization_bits=args.bits
    )
    
    # Export
    converter.export_qubo(formulation, args.output)
    
    print("\n" + "=" * 60)
    print("QUBO FORMULATION COMPLETE")
    print("=" * 60)
