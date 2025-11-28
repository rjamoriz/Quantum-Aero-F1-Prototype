/**
 * MongoDB models for Quantum Optimization (QUBO/QAOA)
 * Integrates synthetic aerodynamic datasets with quantum computing
 */

const mongoose = require('mongoose');

// QUBO Problem Schema
const QUBOProblemSchema = new mongoose.Schema({
  name: { type: String, required: true },
  description: String,
  
  // Source dataset
  source_dataset_id: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'SyntheticDataset',
    required: true,
    index: true
  },
  
  // Problem formulation
  formulation: {
    type: { type: String, enum: ['QUBO', 'QAOA', 'VQE'], required: true },
    
    // Objective function
    objective: {
      type: String,
      enum: ['maximize_L_over_D', 'minimize_drag', 'maximize_downforce', 'balance_optimization', 'multi_objective'],
      required: true
    },
    
    // Constraints
    constraints: [{
      type: { type: String, enum: ['equality', 'inequality'] },
      parameter: String,
      operator: String,  // '==', '<=', '>=', '<', '>'
      value: Number,
      weight: { type: Number, default: 1.0 }
    }],
    
    // Design variables (which geometry parameters to optimize)
    design_variables: [{
      parameter: String,  // e.g., 'main_plane_angle_deg'
      min_value: Number,
      max_value: Number,
      discretization_steps: { type: Number, default: 10 },
      current_value: Number
    }],
    
    // QUBO matrix representation
    qubo_matrix: {
      size: Number,
      matrix: [[Number]],  // Q matrix for QUBO formulation
      offset: { type: Number, default: 0 }
    },
    
    // QAOA specific parameters
    qaoa_params: {
      num_layers: { type: Number, default: 3 },
      mixer_hamiltonian: String,
      cost_hamiltonian: String
    }
  },
  
  // Quantum execution configuration
  quantum_config: {
    backend: {
      type: String,
      enum: ['qiskit_simulator', 'ibm_quantum', 'dwave', 'aws_braket', 'azure_quantum'],
      default: 'qiskit_simulator'
    },
    
    num_qubits: Number,
    num_shots: { type: Number, default: 1000 },
    optimization_level: { type: Number, default: 3, min: 0, max: 3 },
    
    // For real quantum hardware
    device_name: String,
    queue_position: Number,
    
    // Hybrid classical-quantum
    classical_optimizer: {
      type: String,
      enum: ['COBYLA', 'SLSQP', 'ADAM', 'SPSA'],
      default: 'COBYLA'
    },
    max_iterations: { type: Number, default: 100 }
  },
  
  // Execution status
  status: {
    type: String,
    enum: ['created', 'formulating', 'ready', 'queued', 'running', 'completed', 'failed', 'cancelled'],
    default: 'created',
    index: true
  },
  
  // Results
  results: {
    optimal_solution: {
      geometry_params: mongoose.Schema.Types.Mixed,
      predicted_performance: {
        CL: Number,
        CD: Number,
        L_over_D: Number,
        downforce_front: Number,
        downforce_rear: Number,
        balance: Number
      },
      qubo_energy: Number,
      confidence: Number
    },
    
    // Top N solutions
    pareto_front: [{
      geometry_params: mongoose.Schema.Types.Mixed,
      performance: mongoose.Schema.Types.Mixed,
      rank: Number
    }],
    
    // Convergence data
    convergence_history: [{
      iteration: Number,
      energy: Number,
      parameters: [Number],
      timestamp: Date
    }],
    
    // Quantum metrics
    quantum_metrics: {
      circuit_depth: Number,
      gate_count: Number,
      execution_time_seconds: Number,
      quantum_volume: Number,
      fidelity: Number
    },
    
    // Comparison with classical
    classical_baseline: {
      method: String,
      best_L_over_D: Number,
      execution_time_seconds: Number
    },
    
    improvement_over_classical: Number  // percentage
  },
  
  // Execution tracking
  execution: {
    job_id: String,  // Quantum backend job ID
    started_at: Date,
    completed_at: Date,
    total_runtime_seconds: Number,
    
    // Cost tracking (for cloud quantum services)
    estimated_cost_usd: Number,
    actual_cost_usd: Number
  },
  
  // Validation
  validation: {
    validated: { type: Boolean, default: false },
    validation_method: String,  // 'VLM', 'CFD', 'wind_tunnel'
    actual_performance: mongoose.Schema.Types.Mixed,
    error_percentage: Number
  },
  
  // Metadata
  created_by: String,
  created_at: { type: Date, default: Date.now },
  tags: [String],
  
  // Error tracking
  error: {
    message: String,
    stack: String,
    timestamp: Date
  }
});

// Optimization Campaign Schema (for managing multiple optimization runs)
const OptimizationCampaignSchema = new mongoose.Schema({
  name: { type: String, required: true },
  description: String,
  
  // Campaign configuration
  campaign_type: {
    type: String,
    enum: ['single_point', 'multi_point', 'robust', 'sensitivity_analysis'],
    default: 'single_point'
  },
  
  // Associated problems
  optimization_problems: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'QUBOProblem'
  }],
  
  // Campaign-level objectives
  objectives: [{
    name: String,
    weight: Number,
    target_value: Number
  }],
  
  // Operating conditions
  operating_conditions: [{
    name: String,  // e.g., 'high_speed_straight', 'low_speed_corner'
    V_inf: Number,
    yaw: Number,
    ground_gap: Number,
    weight: Number  // importance weight
  }],
  
  // Campaign status
  status: {
    type: String,
    enum: ['planning', 'running', 'completed', 'failed'],
    default: 'planning'
  },
  
  // Aggregated results
  campaign_results: {
    best_overall_design: mongoose.Schema.Types.Mixed,
    performance_across_conditions: [mongoose.Schema.Types.Mixed],
    robustness_score: Number,
    
    // Trade-off analysis
    pareto_optimal_designs: [{
      design: mongoose.Schema.Types.Mixed,
      objectives: mongoose.Schema.Types.Mixed,
      rank: Number
    }]
  },
  
  // Progress tracking
  progress: {
    total_optimizations: Number,
    completed_optimizations: Number,
    percentage: Number
  },
  
  created_at: { type: Date, default: Date.now },
  completed_at: Date,
  created_by: String
});

// Surrogate Model Schema (ML model trained on synthetic data for fast predictions)
const SurrogateModelSchema = new mongoose.Schema({
  name: { type: String, required: true },
  description: String,
  
  // Training dataset
  training_dataset_id: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'SyntheticDataset',
    required: true
  },
  
  // Model architecture
  model_type: {
    type: String,
    enum: ['neural_network', 'gaussian_process', 'random_forest', 'gradient_boosting', 'transformer'],
    required: true
  },
  
  architecture: {
    input_features: [String],  // List of geometry parameters
    output_targets: [String],  // ['CL', 'CD', 'L_over_D', etc.]
    hidden_layers: [Number],
    activation: String,
    total_parameters: Number
  },
  
  // Training metrics
  training_metrics: {
    train_loss: Number,
    val_loss: Number,
    test_loss: Number,
    
    r2_score: mongoose.Schema.Types.Mixed,  // Per output
    mae: mongoose.Schema.Types.Mixed,
    rmse: mongoose.Schema.Types.Mixed,
    
    training_time_seconds: Number,
    epochs: Number
  },
  
  // Model storage
  model_path: String,  // Path to saved model file
  model_format: String,  // 'pytorch', 'tensorflow', 'onnx', 'pickle'
  
  // Inference performance
  inference_metrics: {
    avg_inference_time_ms: Number,
    throughput_samples_per_second: Number
  },
  
  // Status
  status: {
    type: String,
    enum: ['training', 'trained', 'deployed', 'failed'],
    default: 'training'
  },
  
  // Deployment
  deployment: {
    endpoint_url: String,
    deployed_at: Date,
    version: String
  },
  
  created_at: { type: Date, default: Date.now },
  updated_at: Date
});

// Indexes for performance
QUBOProblemSchema.index({ status: 1, created_at: -1 });
QUBOProblemSchema.index({ source_dataset_id: 1 });
QUBOProblemSchema.index({ 'formulation.objective': 1 });

OptimizationCampaignSchema.index({ status: 1, created_at: -1 });
SurrogateModelSchema.index({ status: 1, training_dataset_id: 1 });

// Methods
QUBOProblemSchema.methods.updateStatus = function(status, error = null) {
  this.status = status;
  if (error) {
    this.error = {
      message: error.message,
      stack: error.stack,
      timestamp: new Date()
    };
  }
  return this.save();
};

QUBOProblemSchema.methods.recordResult = function(result) {
  this.results.optimal_solution = result.optimal_solution;
  this.results.pareto_front = result.pareto_front || [];
  this.results.convergence_history = result.convergence_history || [];
  this.results.quantum_metrics = result.quantum_metrics;
  
  if (result.classical_baseline) {
    this.results.classical_baseline = result.classical_baseline;
    
    // Calculate improvement
    const quantum_ld = result.optimal_solution.predicted_performance.L_over_D;
    const classical_ld = result.classical_baseline.best_L_over_D;
    this.results.improvement_over_classical = ((quantum_ld - classical_ld) / classical_ld) * 100;
  }
  
  this.status = 'completed';
  this.execution.completed_at = new Date();
  this.execution.total_runtime_seconds = 
    (this.execution.completed_at - this.execution.started_at) / 1000;
  
  return this.save();
};

const QUBOProblem = mongoose.model('QUBOProblem', QUBOProblemSchema);
const OptimizationCampaign = mongoose.model('OptimizationCampaign', OptimizationCampaignSchema);
const SurrogateModel = mongoose.model('SurrogateModel', SurrogateModelSchema);

module.exports = {
  QUBOProblem,
  OptimizationCampaign,
  SurrogateModel
};
