"""
Classical Simulated Annealing Optimizer
Fallback for quantum optimization when quantum resources unavailable
"""

import numpy as np
from typing import Dict, Callable, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnnealingResult:
    """Simulated annealing result"""
    solution: np.ndarray
    cost: float
    iterations: int
    success: bool
    method: str = "Simulated Annealing"


class SimulatedAnnealing:
    """
    Classical simulated annealing optimizer.
    
    Provides baseline for quantum optimization comparison.
    Often performs well on discrete optimization problems.
    """
    
    def __init__(
        self,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        max_iterations: int = 1000,
        min_temp: float = 0.01
    ):
        """
        Initialize simulated annealing optimizer.
        
        Args:
            initial_temp: Starting temperature
            cooling_rate: Temperature reduction factor (0 < rate < 1)
            max_iterations: Maximum iterations
            min_temp: Minimum temperature (stopping criterion)
        """
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.min_temp = min_temp
        
        logger.info(f"Simulated Annealing initialized: T0={initial_temp}, rate={cooling_rate}")
    
    def optimize(
        self,
        cost_function: Callable,
        n_variables: int,
        constraints: Optional[Dict] = None,
        initial_solution: Optional[np.ndarray] = None
    ) -> AnnealingResult:
        """
        Optimize using simulated annealing.
        
        Args:
            cost_function: Function to minimize f(x) where x ∈ {0,1}^n
            n_variables: Number of binary variables
            constraints: Optional constraints
            initial_solution: Starting solution (random if None)
            
        Returns:
            AnnealingResult with optimal solution
        """
        logger.info(f"Starting simulated annealing: {n_variables} variables")
        
        # Initialize solution
        if initial_solution is not None:
            current_solution = initial_solution.copy()
        else:
            current_solution = np.random.randint(0, 2, n_variables)
        
        current_cost = cost_function(current_solution)
        
        # Track best solution
        best_solution = current_solution.copy()
        best_cost = current_cost
        
        # Annealing schedule
        temperature = self.initial_temp
        iteration = 0
        
        while temperature > self.min_temp and iteration < self.max_iterations:
            # Generate neighbor (flip random bit)
            neighbor = current_solution.copy()
            flip_idx = np.random.randint(0, n_variables)
            neighbor[flip_idx] = 1 - neighbor[flip_idx]
            
            # Check constraints
            if constraints is not None and not self._check_constraints(neighbor, constraints):
                continue
            
            # Evaluate neighbor
            neighbor_cost = cost_function(neighbor)
            
            # Accept or reject
            delta = neighbor_cost - current_cost
            
            if delta < 0:
                # Better solution - always accept
                current_solution = neighbor
                current_cost = neighbor_cost
                
                # Update best
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
            else:
                # Worse solution - accept with probability
                acceptance_prob = np.exp(-delta / temperature)
                if np.random.rand() < acceptance_prob:
                    current_solution = neighbor
                    current_cost = neighbor_cost
            
            # Cool down
            temperature *= self.cooling_rate
            iteration += 1
            
            # Log progress
            if iteration % 100 == 0:
                logger.debug(f"Iteration {iteration}: T={temperature:.4f}, best_cost={best_cost:.4f}")
        
        logger.info(f"Annealing completed: cost={best_cost:.4f}, iterations={iteration}")
        
        return AnnealingResult(
            solution=best_solution,
            cost=best_cost,
            iterations=iteration,
            success=True,
            method="Simulated Annealing"
        )
    
    def optimize_qubo(
        self,
        qubo_matrix: np.ndarray,
        constraints: Optional[Dict] = None
    ) -> AnnealingResult:
        """
        Optimize QUBO problem using simulated annealing.
        
        Args:
            qubo_matrix: Q matrix (n x n)
            constraints: Optional constraints
            
        Returns:
            AnnealingResult
        """
        n_variables = qubo_matrix.shape[0]
        
        # Define cost function
        def cost_function(x):
            return x @ qubo_matrix @ x
        
        return self.optimize(cost_function, n_variables, constraints)
    
    def _check_constraints(self, solution: np.ndarray, constraints: Dict) -> bool:
        """Check if solution satisfies constraints"""
        # One-hot constraints
        if 'one_hot_groups' in constraints:
            for group in constraints['one_hot_groups']:
                if solution[group].sum() != 1:
                    return False
        
        return True


class GeneticAlgorithm:
    """
    Genetic Algorithm optimizer for discrete problems.
    Alternative classical method for comparison.
    """
    
    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        max_generations: int = 100
    ):
        """
        Initialize genetic algorithm.
        
        Args:
            population_size: Number of individuals in population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            max_generations: Maximum generations
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        
        logger.info(f"Genetic Algorithm initialized: pop={population_size}, gen={max_generations}")
    
    def optimize(
        self,
        cost_function: Callable,
        n_variables: int,
        constraints: Optional[Dict] = None
    ) -> AnnealingResult:
        """
        Optimize using genetic algorithm.
        
        Args:
            cost_function: Function to minimize
            n_variables: Number of binary variables
            constraints: Optional constraints
            
        Returns:
            AnnealingResult with optimal solution
        """
        logger.info(f"Starting genetic algorithm: {n_variables} variables")
        
        # Initialize population
        population = np.random.randint(0, 2, (self.population_size, n_variables))
        
        best_solution = None
        best_cost = float('inf')
        
        for generation in range(self.max_generations):
            # Evaluate fitness
            fitness = np.array([cost_function(ind) for ind in population])
            
            # Track best
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_cost:
                best_cost = fitness[min_idx]
                best_solution = population[min_idx].copy()
            
            # Selection (tournament)
            new_population = []
            for _ in range(self.population_size):
                # Tournament selection
                idx1, idx2 = np.random.choice(self.population_size, 2, replace=False)
                winner = idx1 if fitness[idx1] < fitness[idx2] else idx2
                new_population.append(population[winner].copy())
            
            population = np.array(new_population)
            
            # Crossover
            for i in range(0, self.population_size-1, 2):
                if np.random.rand() < self.crossover_rate:
                    point = np.random.randint(1, n_variables)
                    population[i, point:], population[i+1, point:] = \
                        population[i+1, point:].copy(), population[i, point:].copy()
            
            # Mutation
            for i in range(self.population_size):
                for j in range(n_variables):
                    if np.random.rand() < self.mutation_rate:
                        population[i, j] = 1 - population[i, j]
            
            if generation % 10 == 0:
                logger.debug(f"Generation {generation}: best_cost={best_cost:.4f}")
        
        logger.info(f"GA completed: cost={best_cost:.4f}")
        
        return AnnealingResult(
            solution=best_solution,
            cost=best_cost,
            iterations=self.max_generations,
            success=True,
            method="Genetic Algorithm"
        )


class HybridOptimizer:
    """
    Hybrid optimizer that selects best method based on problem size.
    """
    
    def __init__(self):
        self.sa = SimulatedAnnealing()
        self.ga = GeneticAlgorithm()
    
    def optimize(
        self,
        cost_function: Callable,
        n_variables: int,
        constraints: Optional[Dict] = None
    ) -> AnnealingResult:
        """
        Select and run best optimization method.
        
        Args:
            cost_function: Function to minimize
            n_variables: Number of variables
            constraints: Optional constraints
            
        Returns:
            Best result from available methods
        """
        # For small problems, use simulated annealing
        if n_variables <= 30:
            logger.info("Using Simulated Annealing (small problem)")
            return self.sa.optimize(cost_function, n_variables, constraints)
        
        # For larger problems, use genetic algorithm
        else:
            logger.info("Using Genetic Algorithm (large problem)")
            return self.ga.optimize(cost_function, n_variables, constraints)


if __name__ == "__main__":
    # Test classical optimizers
    logging.basicConfig(level=logging.INFO)
    
    # Simple QUBO problem
    Q = np.array([
        [1, -2, 0],
        [-2, 2, -1],
        [0, -1, 1]
    ])
    
    print("Classical Optimizer Test")
    print("=" * 50)
    print(f"QUBO matrix:\n{Q}")
    
    # Test Simulated Annealing
    sa = SimulatedAnnealing()
    sa_result = sa.optimize_qubo(Q)
    print(f"\nSimulated Annealing:")
    print(f"  Solution: {sa_result.solution}")
    print(f"  Cost: {sa_result.cost:.4f}")
    
    # Test Genetic Algorithm
    ga = GeneticAlgorithm()
    def cost_fn(x): return x @ Q @ x
    ga_result = ga.optimize(cost_fn, 3)
    print(f"\nGenetic Algorithm:")
    print(f"  Solution: {ga_result.solution}")
    print(f"  Cost: {ga_result.cost:.4f}")
