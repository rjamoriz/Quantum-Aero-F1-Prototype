"""
Quantum Data Encoding/Decoding
Convert aerodynamic data to quantum-compatible formats
For IBM Qiskit and D-Wave Ocean SDK
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class QuantumDataEncoder:
    """
    Encode aerodynamic optimization problems for quantum computers
    
    Supports:
    - QUBO encoding for D-Wave
    - Hamiltonian encoding for IBM Quantum (VQE)
    - Parameter discretization
    - Constraint handling
    """
    
    def __init__(self, num_bits_per_variable: int = 4):
        self.num_bits_per_variable = num_bits_per_variable
        self.max_value = 2**num_bits_per_variable - 1
    
    def encode_aerodynamic_parameters(
        self,
        parameters: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]]
    ) -> Dict[str, List[int]]:
        """
        Encode continuous aerodynamic parameters to binary
        
        Args:
            parameters: Dict of parameter names to values
            bounds: Dict of parameter names to (min, max) bounds
        
        Returns:
            Dict of parameter names to binary encodings
        """
        encoded = {}
        
        for param_name, value in parameters.items():
            if param_name in bounds:
                min_val, max_val = bounds[param_name]
                
                # Normalize to [0, 1]
                normalized = (value - min_val) / (max_val - min_val)
                normalized = np.clip(normalized, 0, 1)
                
                # Convert to integer
                int_value = int(normalized * self.max_value)
                
                # Convert to binary
                binary = [int(b) for b in format(int_value, f'0{self.num_bits_per_variable}b')]
                
                encoded[param_name] = binary
        
        return encoded
    
    def decode_binary_to_parameters(
        self,
        binary_encoding: Dict[str, List[int]],
        bounds: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Decode binary encoding back to continuous parameters
        
        Args:
            binary_encoding: Dict of parameter names to binary lists
            bounds: Dict of parameter names to (min, max) bounds
        
        Returns:
            Dict of parameter names to decoded values
        """
        decoded = {}
        
        for param_name, binary in binary_encoding.items():
            if param_name in bounds:
                # Convert binary to integer
                int_value = int(''.join(map(str, binary)), 2)
                
                # Normalize
                normalized = int_value / self.max_value
                
                # Scale to bounds
                min_val, max_val = bounds[param_name]
                value = min_val + normalized * (max_val - min_val)
                
                decoded[param_name] = value
        
        return decoded
    
    def create_qubo_matrix(
        self,
        objective_coefficients: np.ndarray,
        constraint_matrix: Optional[np.ndarray] = None,
        penalty_strength: float = 10.0
    ) -> np.ndarray:
        """
        Create QUBO matrix for D-Wave annealing
        
        QUBO form: minimize x^T Q x
        
        Args:
            objective_coefficients: Linear objective coefficients
            constraint_matrix: Constraint coefficients (optional)
            penalty_strength: Penalty for constraint violations
        
        Returns:
            QUBO matrix Q
        """
        n = len(objective_coefficients)
        Q = np.zeros((n, n))
        
        # Add objective (diagonal terms)
        for i in range(n):
            Q[i, i] = objective_coefficients[i]
        
        # Add constraints as penalties
        if constraint_matrix is not None:
            # Add quadratic penalty terms
            Q += penalty_strength * (constraint_matrix.T @ constraint_matrix)
        
        return Q
    
    def aerodynamic_to_qubo(
        self,
        num_design_vars: int,
        target_cl: float = 2.8,
        target_cd: float = 0.4,
        cl_weight: float = 1.0,
        cd_weight: float = 0.5
    ) -> np.ndarray:
        """
        Convert aerodynamic optimization to QUBO
        
        Objective: Maximize Cl, minimize Cd
        
        Args:
            num_design_vars: Number of design variables
            target_cl: Target lift coefficient
            target_cd: Target drag coefficient
            cl_weight: Weight for lift objective
            cd_weight: Weight for drag objective
        
        Returns:
            QUBO matrix
        """
        # Linear coefficients
        # Negative for maximization (Cl)
        # Positive for minimization (Cd)
        coefficients = np.zeros(num_design_vars)
        
        for i in range(num_design_vars):
            # Downforce contribution (negative = maximize)
            coefficients[i] = -cl_weight * (1 + i / num_design_vars)
            
            # Drag penalty (positive = minimize)
            coefficients[i] += cd_weight * (0.5 + i / num_design_vars)
        
        # Create QUBO
        Q = self.create_qubo_matrix(coefficients)
        
        # Add interaction terms (aerodynamic coupling)
        for i in range(num_design_vars - 1):
            for j in range(i + 1, min(i + 5, num_design_vars)):
                # Adjacent variables interact
                Q[i, j] = 0.1 * np.random.randn()
                Q[j, i] = Q[i, j]  # Symmetric
        
        return Q
    
    def create_ising_hamiltonian(
        self,
        qubo_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convert QUBO to Ising Hamiltonian for VQE
        
        QUBO: x ∈ {0,1}
        Ising: z ∈ {-1,1}
        
        Transformation: x_i = (1 - z_i) / 2
        
        Returns:
            (h, J, offset) where H = Σ h_i z_i + Σ J_ij z_i z_j + offset
        """
        n = qubo_matrix.shape[0]
        
        # Linear terms (h)
        h = np.zeros(n)
        for i in range(n):
            h[i] = -qubo_matrix[i, i] / 2
            for j in range(n):
                if i != j:
                    h[i] -= qubo_matrix[i, j] / 4
        
        # Quadratic terms (J)
        J = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                J[i, j] = qubo_matrix[i, j] / 4
        
        # Offset
        offset = np.sum(qubo_matrix) / 4
        
        return h, J, offset


class QuantumResultDecoder:
    """
    Decode quantum optimization results back to aerodynamic parameters
    """
    
    def __init__(self):
        pass
    
    def decode_dwave_solution(
        self,
        solution: Dict[int, int],
        parameter_mapping: Dict[str, List[int]],
        bounds: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Decode D-Wave solution to aerodynamic parameters
        
        Args:
            solution: Binary solution from D-Wave
            parameter_mapping: Mapping of parameters to variable indices
            bounds: Parameter bounds
        
        Returns:
            Decoded aerodynamic parameters
        """
        encoder = QuantumDataEncoder()
        
        # Extract binary values for each parameter
        binary_encoding = {}
        for param_name, indices in parameter_mapping.items():
            binary = [solution.get(idx, 0) for idx in indices]
            binary_encoding[param_name] = binary
        
        # Decode to continuous values
        parameters = encoder.decode_binary_to_parameters(binary_encoding, bounds)
        
        return parameters
    
    def decode_vqe_solution(
        self,
        measurement_counts: Dict[str, int],
        parameter_mapping: Dict[str, List[int]],
        bounds: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Decode VQE measurement results to aerodynamic parameters
        
        Args:
            measurement_counts: Measurement histogram from VQE
            parameter_mapping: Mapping of parameters to qubit indices
            bounds: Parameter bounds
        
        Returns:
            Decoded aerodynamic parameters
        """
        # Get most probable measurement
        most_probable = max(measurement_counts, key=measurement_counts.get)
        
        # Convert bitstring to solution dict
        solution = {i: int(bit) for i, bit in enumerate(most_probable)}
        
        # Decode using same method as D-Wave
        return self.decode_dwave_solution(solution, parameter_mapping, bounds)
    
    def compute_aerodynamic_coefficients(
        self,
        parameters: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Estimate aerodynamic coefficients from design parameters
        
        Simplified model for demonstration
        """
        # Extract parameters
        angle = parameters.get('angle_of_attack', 0)
        camber = parameters.get('camber', 0.04)
        thickness = parameters.get('thickness', 0.12)
        
        # Simplified aerodynamic model
        cl = 0.1 * angle + 10 * camber
        cd = 0.01 + 0.001 * angle**2 + 0.05 * thickness
        cm = -0.1 * camber
        
        return {
            'cl': cl,
            'cd': cd,
            'cm': cm,
            'l_d_ratio': cl / cd if cd > 0 else 0
        }


# Example usage
if __name__ == "__main__":
    print("Quantum Data Encoding Example\n")
    
    # Define aerodynamic parameters
    parameters = {
        'angle_of_attack': 5.0,
        'camber': 0.04,
        'thickness': 0.12,
        'flap_angle': 10.0
    }
    
    bounds = {
        'angle_of_attack': (-10.0, 20.0),
        'camber': (0.0, 0.08),
        'thickness': (0.08, 0.16),
        'flap_angle': (0.0, 30.0)
    }
    
    # Encode
    encoder = QuantumDataEncoder(num_bits_per_variable=4)
    encoded = encoder.encode_aerodynamic_parameters(parameters, bounds)
    
    print("Encoded parameters:")
    for param, binary in encoded.items():
        print(f"  {param}: {binary} (decimal: {int(''.join(map(str, binary)), 2)})")
    
    # Decode
    decoded = encoder.decode_binary_to_parameters(encoded, bounds)
    
    print("\nDecoded parameters:")
    for param, value in decoded.items():
        original = parameters[param]
        error = abs(value - original)
        print(f"  {param}: {value:.4f} (original: {original:.4f}, error: {error:.4f})")
    
    # Create QUBO
    print("\nCreating QUBO matrix...")
    qubo = encoder.aerodynamic_to_qubo(num_design_vars=16)
    print(f"  QUBO shape: {qubo.shape}")
    print(f"  Non-zero elements: {np.count_nonzero(qubo)}")
    
    # Convert to Ising
    print("\nConverting to Ising Hamiltonian...")
    h, J, offset = encoder.create_ising_hamiltonian(qubo)
    print(f"  Linear terms (h): {h.shape}")
    print(f"  Quadratic terms (J): {J.shape}")
    print(f"  Offset: {offset:.4f}")
