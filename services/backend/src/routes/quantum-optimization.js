/**
 * Quantum Optimization API Routes
 * Integrates synthetic datasets with QUBO/QAOA quantum optimization
 */

const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;

const { QUBOProblem, OptimizationCampaign, SurrogateModel } = require('../models/QuantumOptimization');
const { SyntheticDataset } = require('../models/SyntheticDataset');

// ============================================================================
// QUBO PROBLEM ROUTES
// ============================================================================

/**
 * GET /api/quantum/qubo-problems
 * List all QUBO problems
 */
router.get('/qubo-problems', async (req, res) => {
  try {
    const { status, objective, limit = 20, skip = 0 } = req.query;
    
    const query = {};
    if (status) query.status = status;
    if (objective) query['formulation.objective'] = objective;
    
    const problems = await QUBOProblem.find(query)
      .populate('source_dataset_id', 'name description statistics')
      .sort({ created_at: -1 })
      .limit(parseInt(limit))
      .skip(parseInt(skip));
    
    const total = await QUBOProblem.countDocuments(query);
    
    res.json({
      success: true,
      data: problems,
      pagination: {
        total,
        limit: parseInt(limit),
        skip: parseInt(skip),
        hasMore: (parseInt(skip) + parseInt(limit)) < total
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * POST /api/quantum/qubo-problems
 * Create new QUBO problem from dataset
 */
router.post('/qubo-problems', async (req, res) => {
  try {
    const {
      name,
      description,
      source_dataset_id,
      objective = 'maximize_L_over_D',
      design_variables,
      constraints = [],
      quantum_backend = 'qiskit_simulator',
      num_shots = 1000,
      discretization_bits = 4
    } = req.body;
    
    // Validate dataset exists
    const dataset = await SyntheticDataset.findById(source_dataset_id);
    if (!dataset) {
      return res.status(404).json({ success: false, error: 'Dataset not found' });
    }
    
    if (dataset.status !== 'completed') {
      return res.status(400).json({ 
        success: false, 
        error: 'Dataset must be completed before creating QUBO problem' 
      });
    }
    
    // Default design variables if not specified
    const defaultVariables = design_variables || [
      { parameter: 'main_plane_angle_deg', min_value: -5, max_value: 15, discretization_steps: 10 },
      { parameter: 'rear_wing_angle_deg', min_value: 0, max_value: 30, discretization_steps: 10 },
      { parameter: 'floor_gap', min_value: 0.01, max_value: 0.05, discretization_steps: 10 }
    ];
    
    // Calculate num_qubits
    const num_qubits = defaultVariables.length * discretization_bits;
    
    // Create QUBO problem
    const problem = new QUBOProblem({
      name,
      description,
      source_dataset_id,
      formulation: {
        type: 'QUBO',
        objective,
        constraints,
        design_variables: defaultVariables,
        qubo_matrix: {
          size: num_qubits,
          matrix: [],  // Will be populated during formulation
          offset: 0
        }
      },
      quantum_config: {
        backend: quantum_backend,
        num_qubits,
        num_shots,
        optimization_level: 3,
        classical_optimizer: 'COBYLA',
        max_iterations: 100
      },
      status: 'created'
    });
    
    await problem.save();
    
    res.json({
      success: true,
      data: problem,
      message: 'QUBO problem created. Use /formulate endpoint to generate QUBO matrix.'
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * POST /api/quantum/qubo-problems/:id/formulate
 * Formulate QUBO matrix from dataset
 */
router.post('/qubo-problems/:id/formulate', async (req, res) => {
  try {
    const problem = await QUBOProblem.findById(req.params.id)
      .populate('source_dataset_id');
    
    if (!problem) {
      return res.status(404).json({ success: false, error: 'Problem not found' });
    }
    
    await problem.updateStatus('formulating');
    
    // Get dataset path
    const dataset = problem.source_dataset_id;
    const scalarsPath = path.join(dataset.storage.output_dir, 'scalars.json');
    
    // Call Python converter
    const pythonScript = path.join(__dirname, '../../../quantum_service/dataset_to_qubo.py');
    const outputPath = path.join(dataset.storage.output_dir, `qubo_${problem._id}.json`);
    
    const args = [
      pythonScript,
      '--dataset', scalarsPath,
      '--objective', problem.formulation.objective,
      '--output', outputPath,
      '--bits', '4'
    ];
    
    const pythonProcess = spawn('python', args);
    
    let stdout = '';
    let stderr = '';
    
    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
      console.log(`QUBO formulation: ${data}`);
    });
    
    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
      console.error(`QUBO error: ${data}`);
    });
    
    pythonProcess.on('close', async (code) => {
      if (code === 0) {
        try {
          // Load QUBO formulation
          const quboData = JSON.parse(await fs.readFile(outputPath, 'utf-8'));
          
          // Update problem
          problem.formulation.qubo_matrix = {
            size: quboData.num_qubits,
            matrix: quboData.Q_matrix,
            offset: quboData.offset
          };
          problem.status = 'ready';
          
          await problem.save();
          
          res.json({
            success: true,
            data: problem,
            message: 'QUBO formulation complete. Ready for quantum execution.'
          });
        } catch (error) {
          await problem.updateStatus('failed', error);
          res.status(500).json({ success: false, error: error.message });
        }
      } else {
        const error = new Error(`QUBO formulation failed: ${stderr}`);
        await problem.updateStatus('failed', error);
        res.status(500).json({ success: false, error: error.message });
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * POST /api/quantum/qubo-problems/:id/execute
 * Execute quantum optimization
 */
router.post('/qubo-problems/:id/execute', async (req, res) => {
  try {
    const problem = await QUBOProblem.findById(req.params.id);
    
    if (!problem) {
      return res.status(404).json({ success: false, error: 'Problem not found' });
    }
    
    if (problem.status !== 'ready') {
      return res.status(400).json({ 
        success: false, 
        error: 'Problem must be formulated before execution' 
      });
    }
    
    await problem.updateStatus('queued');
    problem.execution.started_at = new Date();
    await problem.save();
    
    // Execute quantum optimization asynchronously
    executeQuantumOptimization(problem._id).catch(console.error);
    
    res.json({
      success: true,
      data: problem,
      message: 'Quantum optimization queued. Check status for updates.'
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * GET /api/quantum/qubo-problems/:id
 * Get QUBO problem details
 */
router.get('/qubo-problems/:id', async (req, res) => {
  try {
    const problem = await QUBOProblem.findById(req.params.id)
      .populate('source_dataset_id');
    
    if (!problem) {
      return res.status(404).json({ success: false, error: 'Problem not found' });
    }
    
    res.json({ success: true, data: problem });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * DELETE /api/quantum/qubo-problems/:id
 * Delete QUBO problem
 */
router.delete('/qubo-problems/:id', async (req, res) => {
  try {
    const problem = await QUBOProblem.findByIdAndDelete(req.params.id);
    
    if (!problem) {
      return res.status(404).json({ success: false, error: 'Problem not found' });
    }
    
    res.json({ success: true, message: 'QUBO problem deleted' });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// ============================================================================
// OPTIMIZATION CAMPAIGNS
// ============================================================================

/**
 * POST /api/quantum/campaigns
 * Create optimization campaign
 */
router.post('/campaigns', async (req, res) => {
  try {
    const {
      name,
      description,
      campaign_type = 'single_point',
      objectives,
      operating_conditions
    } = req.body;
    
    const campaign = new OptimizationCampaign({
      name,
      description,
      campaign_type,
      objectives,
      operating_conditions,
      status: 'planning'
    });
    
    await campaign.save();
    
    res.json({
      success: true,
      data: campaign,
      message: 'Campaign created. Add optimization problems to begin.'
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * GET /api/quantum/campaigns
 * List campaigns
 */
router.get('/campaigns', async (req, res) => {
  try {
    const campaigns = await OptimizationCampaign.find()
      .populate('optimization_problems')
      .sort({ created_at: -1 });
    
    res.json({ success: true, data: campaigns });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Execute quantum optimization (async)
 */
async function executeQuantumOptimization(problemId) {
  try {
    const problem = await QUBOProblem.findById(problemId)
      .populate('source_dataset_id');
    
    await problem.updateStatus('running');
    
    // Use existing VQE service or Qiskit QAOA
    const vqeScript = path.join(__dirname, '../../../quantum_service/vqe/optimizer.py');
    const dataset = problem.source_dataset_id;
    const quboPath = path.join(dataset.storage.output_dir, `qubo_${problem._id}.json`);
    const outputPath = path.join(dataset.storage.output_dir, `result_${problem._id}.json`);
    
    const args = [
      vqeScript,
      '--qubo', quboPath,
      '--backend', problem.quantum_config.backend,
      '--shots', problem.quantum_config.num_shots.toString(),
      '--output', outputPath
    ];
    
    const pythonProcess = spawn('python', args);
    
    pythonProcess.stdout.on('data', (data) => {
      console.log(`Quantum execution: ${data}`);
    });
    
    pythonProcess.on('close', async (code) => {
      if (code === 0) {
        // Load results
        const resultData = JSON.parse(await fs.readFile(outputPath, 'utf-8'));
        
        // Record results
        await problem.recordResult({
          optimal_solution: resultData.optimal_solution,
          pareto_front: resultData.pareto_front,
          convergence_history: resultData.convergence_history,
          quantum_metrics: resultData.quantum_metrics,
          classical_baseline: resultData.classical_baseline
        });
        
        console.log(`✓ Quantum optimization ${problemId} completed`);
      } else {
        const error = new Error('Quantum execution failed');
        await problem.updateStatus('failed', error);
        console.error(`✗ Quantum optimization ${problemId} failed`);
      }
    });
  } catch (error) {
    console.error(`Error in quantum optimization ${problemId}:`, error);
    const problem = await QUBOProblem.findById(problemId);
    if (problem) {
      await problem.updateStatus('failed', error);
    }
  }
}

module.exports = router;
