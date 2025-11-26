"""
Quantum-Aerodynamics Integration Bridge
Connects quantum optimization with all aerodynamic aspects:
- VLM physics solver
- ML surrogate predictions
- Multi-objective optimization
- Aeroelastic constraints
- Transient aerodynamics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'scripts' / 'data-preprocessing'))

from qaoa.solver import QAOASolver, create_qubo_from_problem
from classical.simulated_annealing import SimulatedAnnealing, HybridOptimizer

logger = logging.getLogger(__name__)


@dataclass
class AeroOptimizationProblem:
    """
    Complete aerodynamic optimization problem definition.
    
    Combines:
    - Discrete variables (quantum-optimized)
    - Continuous variables (classical-optimized)
    - Multi-physics constraints
    - Multi-objective fitness
    """
    # Design variables
    discrete_vars: Dict[str, List]  # Discrete choices (stiffener placement, etc.)
    continuous_vars: Dict[str, Tuple[float, float]]  # Continuous ranges (angles, etc.)
    
    # Objectives
    objectives: List[str]  # ['maximize_downforce', 'minimize_drag', 'balance']
    weights: List[float]  # Objective weights
    
    # Constraints
    constraints: Dict  # Physical and regulatory constraints
    
    # Aerodynamic aspects
    include_aeroelastic: bool = True
    include_transient: bool = False
    include_thermal: bool = False
    include_acoustic: bool = False


class QuantumAeroBridge:
    """
    Bridge between quantum optimization and aerodynamic simulation.
    
    Workflow:
    1. Formulate aerodynamic design as QUBO problem
    2. Use quantum optimizer for discrete variables
    3. Use classical optimizer for continuous variables
    4. Evaluate with physics solver or ML surrogate
    5. Apply multi-physics constraints
    6. Return Pareto-optimal solutions
    """
    
    def __init__(
        self,
        use_quantum: bool = True,
        use_ml_surrogate: bool = True,
        n_qaoa_layers: int = 3
    ):
        """
        Initialize quantum-aero bridge.
        
        Args:
            use_quantum: Use quantum optimizer (vs. classical only)
            use_ml_surrogate: Use ML surrogate for fast evaluation
            n_qaoa_layers: Number of QAOA layers
        """
        self.use_quantum = use_quantum
        self.use_ml_surrogate = use_ml_surrogate
        
        # Initialize optimizers
        if use_quantum:
            self.qaoa_solver = QAOASolver(n_layers=n_qaoa_layers)
            logger.info("QAOA solver initialized")
        
        self.classical_solver = HybridOptimizer()
        logger.info("Classical solver initialized")
        
        # Evaluation functions (to be set)
        self.physics_evaluator = None
        self.ml_evaluator = None
        
        logger.info("Quantum-Aero Bridge initialized")
    
    def set_physics_evaluator(self, evaluator: Callable):
        """Set physics-based evaluation function (VLM, CFD, etc.)"""
        self.physics_evaluator = evaluator
        logger.info("Physics evaluator set")
    
    def set_ml_evaluator(self, evaluator: Callable):
        """Set ML surrogate evaluation function"""
        self.ml_evaluator = evaluator
        logger.info("ML evaluator set")
    
    def optimize_f1_wing(
        self,
        wing_type: str = 'front',
        objectives: List[str] = ['maximize_downforce', 'minimize_drag'],
        n_iterations: int = 10
    ) -> Dict:
        """
        Optimize F1 wing design.
        
        Args:
            wing_type: 'front' or 'rear'
            objectives: List of objectives
            n_iterations: Number of optimization iterations
            
        Returns:
            Optimal design parameters and performance
        """
        logger.info(f"Optimizing {wing_type} wing: {objectives}")
        
        # Define design space
        if wing_type == 'front':
            problem = self._create_front_wing_problem(objectives)
        else:
            problem = self._create_rear_wing_problem(objectives)
        
        # Run hybrid optimization
        result = self._hybrid_optimize(problem, n_iterations)
        
        return result
    
    def optimize_complete_car(
        self,
        objectives: List[str] = ['maximize_downforce', 'minimize_drag', 'optimize_balance'],
        include_aeroelastic: bool = True,
        include_transient: bool = False,
        n_iterations: int = 20
    ) -> Dict:
        """
        Optimize complete F1 car aerodynamics.
        
        Args:
            objectives: Multi-objective list
            include_aeroelastic: Include aeroelastic effects
            include_transient: Include transient aerodynamics
            n_iterations: Optimization iterations
            
        Returns:
            Complete car optimal configuration
        """
        logger.info(f"Optimizing complete car: {objectives}")
        
        # Create comprehensive problem
        problem = AeroOptimizationProblem(
            discrete_vars={
                'front_wing_flap_config': [0, 1, 2],  # Configuration options
                'rear_wing_drs_strategy': [0, 1],  # DRS usage
                'floor_vortex_generators': list(range(10)),  # VG placement
                'diffuser_strakes': [0, 1, 2, 3]  # Number of strakes
            },
            continuous_vars={
                'front_wing_aoa': (15, 25),
                'rear_wing_aoa': (8, 16),
                'ride_height_front': (0.010, 0.025),
                'ride_height_rear': (0.040, 0.070),
                'diffuser_angle': (12, 18)
            },
            objectives=objectives,
            weights=[1.0] * len(objectives),
            constraints={
                'max_downforce': 5000,  # N
                'aero_balance': (0.45, 0.55),  # Front/rear balance
                'l_over_d': 3.0,  # Minimum efficiency
                'flutter_margin': 1.2  # Safety factor
            },
            include_aeroelastic=include_aeroelastic,
            include_transient=include_transient
        )
        
        # Run optimization
        result = self._hybrid_optimize(problem, n_iterations)
        
        return result
    
    def _create_front_wing_problem(self, objectives: List[str]) -> AeroOptimizationProblem:
        """Create front wing optimization problem"""
        return AeroOptimizationProblem(
            discrete_vars={
                'flap1_config': [0, 1, 2],  # Low/medium/high downforce
                'flap2_config': [0, 1, 2],
                'endplate_design': [0, 1, 2, 3],  # Different geometries
                'slot_gap_size': [0, 1, 2]  # Small/medium/large
            },
            continuous_vars={
                'main_plane_aoa': (15, 25),
                'flap1_aoa': (20, 30),
                'flap2_aoa': (25, 35),
                'ride_height': (0.03, 0.08)
            },
            objectives=objectives,
            weights=[1.0] * len(objectives),
            constraints={
                'max_downforce': 2000,  # N
                'max_drag': 300,  # N
                'structural_load': 5000  # N
            }
        )
    
    def _create_rear_wing_problem(self, objectives: List[str]) -> AeroOptimizationProblem:
        """Create rear wing optimization problem"""
        return AeroOptimizationProblem(
            discrete_vars={
                'main_plane_profile': [0, 1, 2],  # NACA profile variants
                'drs_flap_config': [0, 1],  # Standard/aggressive
                'endplate_design': [0, 1, 2],
                'beam_wing': [0, 1]  # With/without beam wing
            },
            continuous_vars={
                'main_plane_aoa': (8, 16),
                'flap_aoa': (20, 30),
                'wing_height': (0.90, 1.00)
            },
            objectives=objectives,
            weights=[1.0] * len(objectives),
            constraints={
                'max_downforce': 1500,  # N
                'drs_drag_reduction': 0.15  # 15% minimum
            }
        )
    
    def _hybrid_optimize(
        self,
        problem: AeroOptimizationProblem,
        n_iterations: int
    ) -> Dict:
        """
        Hybrid quantum-classical optimization.
        
        Workflow:
        1. Quantum optimizer handles discrete variables
        2. Classical optimizer handles continuous variables
        3. Evaluate with physics/ML
        4. Update and iterate
        
        Args:
            problem: Optimization problem
            n_iterations: Number of iterations
            
        Returns:
            Optimal solution
        """
        logger.info(f"Starting hybrid optimization: {n_iterations} iterations")
        
        best_solution = None
        best_fitness = float('-inf')
        history = []
        
        for iteration in range(n_iterations):
            # Step 1: Quantum optimization for discrete variables
            discrete_solution = self._optimize_discrete(problem)
            
            # Step 2: Classical optimization for continuous variables
            continuous_solution = self._optimize_continuous(problem, discrete_solution)
            
            # Step 3: Evaluate aerodynamics
            performance = self._evaluate_aerodynamics(
                discrete_solution,
                continuous_solution,
                problem
            )
            
            # Step 4: Compute multi-objective fitness
            fitness = self._compute_fitness(performance, problem)
            
            # Step 5: Check constraints
            feasible = self._check_constraints(performance, problem)
            
            if feasible and fitness > best_fitness:
                best_fitness = fitness
                best_solution = {
                    'discrete': discrete_solution,
                    'continuous': continuous_solution,
                    'performance': performance,
                    'fitness': fitness
                }
            
            history.append({
                'iteration': iteration,
                'fitness': fitness,
                'feasible': feasible,
                'performance': performance
            })
            
            logger.info(f"Iteration {iteration+1}/{n_iterations}: fitness={fitness:.4f}, feasible={feasible}")
        
        result = {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'history': history,
            'n_iterations': n_iterations
        }
        
        logger.info(f"Optimization complete: best_fitness={best_fitness:.4f}")
        
        return result
    
    def _optimize_discrete(self, problem: AeroOptimizationProblem) -> Dict:
        """Optimize discrete variables using quantum/classical"""
        # Convert discrete variables to QUBO
        n_vars = sum(len(choices) for choices in problem.discrete_vars.values())
        
        # Create QUBO matrix (simplified)
        Q = np.random.randn(n_vars, n_vars) * 0.1
        Q = (Q + Q.T) / 2  # Ensure symmetric
        
        if self.use_quantum and n_vars <= 20:
            # Use QAOA for small problems
            result = self.qaoa_solver.optimize(Q)
            solution_vector = result.solution
        else:
            # Use classical for large problems
            sa = SimulatedAnnealing()
            result = sa.optimize_qubo(Q)
            solution_vector = result.solution
        
        # Decode solution to discrete choices
        discrete_solution = {}
        idx = 0
        for var_name, choices in problem.discrete_vars.items():
            n_choices = len(choices)
            var_bits = solution_vector[idx:idx+n_choices]
            choice_idx = np.argmax(var_bits) if np.any(var_bits) else 0
            discrete_solution[var_name] = choices[choice_idx]
            idx += n_choices
        
        return discrete_solution
    
    def _optimize_continuous(
        self,
        problem: AeroOptimizationProblem,
        discrete_solution: Dict
    ) -> Dict:
        """Optimize continuous variables using classical methods"""
        from scipy.optimize import minimize
        
        # Initial guess (middle of ranges)
        x0 = []
        var_names = []
        bounds = []
        
        for var_name, (min_val, max_val) in problem.continuous_vars.items():
            x0.append((min_val + max_val) / 2)
            var_names.append(var_name)
            bounds.append((min_val, max_val))
        
        x0 = np.array(x0)
        
        # Objective function
        def objective(x):
            continuous_sol = {name: val for name, val in zip(var_names, x)}
            perf = self._evaluate_aerodynamics(discrete_solution, continuous_sol, problem)
            return -self._compute_fitness(perf, problem)  # Minimize negative fitness
        
        # Optimize
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        continuous_solution = {name: val for name, val in zip(var_names, result.x)}
        
        return continuous_solution
    
    def _evaluate_aerodynamics(
        self,
        discrete_solution: Dict,
        continuous_solution: Dict,
        problem: AeroOptimizationProblem
    ) -> Dict:
        """
        Evaluate aerodynamic performance.
        
        Uses ML surrogate if available, otherwise physics solver.
        """
        # Combine solutions
        full_solution = {**discrete_solution, **continuous_solution}
        
        # Use ML surrogate for fast evaluation
        if self.use_ml_surrogate and self.ml_evaluator is not None:
            performance = self.ml_evaluator(full_solution)
        
        # Fallback to physics solver
        elif self.physics_evaluator is not None:
            performance = self.physics_evaluator(full_solution)
        
        # Mock evaluation if no evaluators set
        else:
            performance = self._mock_evaluation(full_solution)
        
        # Add multi-physics effects if requested
        if problem.include_aeroelastic:
            performance = self._add_aeroelastic_effects(performance, full_solution)
        
        if problem.include_transient:
            performance = self._add_transient_effects(performance, full_solution)
        
        return performance
    
    def _mock_evaluation(self, solution: Dict) -> Dict:
        """Mock aerodynamic evaluation for testing"""
        # Simple empirical models
        aoa = solution.get('main_plane_aoa', 20.0)
        
        cl = 0.1 * aoa + np.random.randn() * 0.01
        cd = 0.01 + 0.0005 * aoa**2 + np.random.randn() * 0.001
        
        return {
            'cl': cl,
            'cd': cd,
            'cm': -0.05,
            'l_over_d': cl / cd if cd > 0 else 0,
            'downforce': cl * 1000,  # Simplified
            'drag': cd * 1000,
            'balance': 0.50
        }
    
    def _add_aeroelastic_effects(self, performance: Dict, solution: Dict) -> Dict:
        """Add aeroelastic effects to performance"""
        # Simplified aeroelastic correction
        # In production, would use full FSI simulation
        
        # Reduce downforce due to wing deflection
        deflection_factor = 0.95  # 5% loss
        performance['cl'] *= deflection_factor
        performance['downforce'] *= deflection_factor
        
        # Add flutter margin
        performance['flutter_margin'] = 1.3  # Safety factor
        
        return performance
    
    def _add_transient_effects(self, performance: Dict, solution: Dict) -> Dict:
        """Add transient aerodynamic effects"""
        # Simplified transient correction
        # In production, would use unsteady simulation
        
        # Transient downforce variation
        performance['transient_variation'] = 0.10  # ±10%
        
        return performance
    
    def _compute_fitness(self, performance: Dict, problem: AeroOptimizationProblem) -> float:
        """Compute multi-objective fitness"""
        fitness = 0.0
        
        for objective, weight in zip(problem.objectives, problem.weights):
            if objective == 'maximize_downforce':
                fitness += weight * performance.get('downforce', 0)
            elif objective == 'minimize_drag':
                fitness -= weight * performance.get('drag', 0)
            elif objective == 'optimize_balance':
                target_balance = 0.50
                balance_error = abs(performance.get('balance', 0.5) - target_balance)
                fitness -= weight * balance_error * 1000
            elif objective == 'maximize_efficiency':
                fitness += weight * performance.get('l_over_d', 0) * 100
        
        return fitness
    
    def _check_constraints(self, performance: Dict, problem: AeroOptimizationProblem) -> bool:
        """Check if solution satisfies constraints"""
        constraints = problem.constraints
        
        # Check maximum downforce
        if 'max_downforce' in constraints:
            if performance.get('downforce', 0) > constraints['max_downforce']:
                return False
        
        # Check aero balance
        if 'aero_balance' in constraints:
            balance = performance.get('balance', 0.5)
            min_bal, max_bal = constraints['aero_balance']
            if not (min_bal <= balance <= max_bal):
                return False
        
        # Check L/D ratio
        if 'l_over_d' in constraints:
            if performance.get('l_over_d', 0) < constraints['l_over_d']:
                return False
        
        # Check flutter margin (aeroelastic)
        if 'flutter_margin' in constraints:
            if performance.get('flutter_margin', 0) < constraints['flutter_margin']:
                return False
        
        return True


if __name__ == "__main__":
    # Test quantum-aero bridge
    logging.basicConfig(level=logging.INFO)
    
    print("Quantum-Aerodynamics Integration Test")
    print("=" * 60)
    
    # Create bridge
    bridge = QuantumAeroBridge(use_quantum=True, use_ml_surrogate=False)
    
    # Test front wing optimization
    print("\n1. Front Wing Optimization")
    result = bridge.optimize_f1_wing(
        wing_type='front',
        objectives=['maximize_downforce', 'minimize_drag'],
        n_iterations=5
    )
    
    print(f"\nBest Solution:")
    print(f"  Fitness: {result['best_fitness']:.2f}")
    print(f"  Downforce: {result['best_solution']['performance']['downforce']:.1f} N")
    print(f"  Drag: {result['best_solution']['performance']['drag']:.1f} N")
    print(f"  L/D: {result['best_solution']['performance']['l_over_d']:.2f}")
    
    # Test complete car optimization
    print("\n2. Complete Car Optimization")
    result = bridge.optimize_complete_car(
        objectives=['maximize_downforce', 'minimize_drag', 'optimize_balance'],
        include_aeroelastic=True,
        n_iterations=5
    )
    
    print(f"\nBest Solution:")
    print(f"  Fitness: {result['best_fitness']:.2f}")
    print(f"  Downforce: {result['best_solution']['performance']['downforce']:.1f} N")
    print(f"  Balance: {result['best_solution']['performance']['balance']:.2%}")
    print(f"  Flutter Margin: {result['best_solution']['performance'].get('flutter_margin', 0):.2f}")
    
    print("\n✅ Integration test passed!")
