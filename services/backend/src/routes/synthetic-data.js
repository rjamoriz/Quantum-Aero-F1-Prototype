/**
 * API routes for Synthetic Aerodynamic Dataset Generation
 */

const express = require('express');
const router = express.Router();
const { SyntheticDataset, AeroSample } = require('../models/SyntheticDataset');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const logger = require('../utils/logger');

// Store active generation processes
const activeGenerations = new Map();

/**
 * GET /api/synthetic-data/datasets
 * Get all datasets with optional filters
 */
router.get('/datasets', async (req, res) => {
  try {
    const { status, limit = 20, skip = 0 } = req.query;
    
    const query = status ? { status } : {};
    
    const datasets = await SyntheticDataset.find(query)
      .sort({ created_at: -1 })
      .limit(parseInt(limit))
      .skip(parseInt(skip))
      .select('-error.stack');
    
    const total = await SyntheticDataset.countDocuments(query);
    
    res.json({
      success: true,
      data: datasets,
      pagination: {
        total,
        limit: parseInt(limit),
        skip: parseInt(skip),
        hasMore: total > (parseInt(skip) + parseInt(limit))
      }
    });
  } catch (error) {
    logger.error('Error fetching datasets:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * GET /api/synthetic-data/datasets/:id
 * Get specific dataset by ID
 */
router.get('/datasets/:id', async (req, res) => {
  try {
    const dataset = await SyntheticDataset.findById(req.params.id);
    
    if (!dataset) {
      return res.status(404).json({
        success: false,
        error: 'Dataset not found'
      });
    }
    
    res.json({
      success: true,
      data: dataset
    });
  } catch (error) {
    logger.error('Error fetching dataset:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * POST /api/synthetic-data/datasets
 * Create new dataset generation job
 */
router.post('/datasets', async (req, res) => {
  try {
    const {
      name,
      description,
      tier1_samples = 100,
      tier2_samples = 0,
      tier3_samples = 0,
      workers = 4,
      sampling_method = 'lhs',
      tags = []
    } = req.body;
    
    // Validate input
    if (!name) {
      return res.status(400).json({
        success: false,
        error: 'Dataset name is required'
      });
    }
    
    if (tier1_samples + tier2_samples + tier3_samples === 0) {
      return res.status(400).json({
        success: false,
        error: 'At least one tier must have samples > 0'
      });
    }
    
    // Create output directory
    const outputDir = path.join(
      __dirname,
      '../../../..',
      'data',
      'synthetic_datasets',
      `dataset_${Date.now()}`
    );
    
    // Create dataset record
    const dataset = new SyntheticDataset({
      name,
      description,
      config: {
        tier1_samples,
        tier2_samples,
        tier3_samples,
        workers,
        sampling_method
      },
      storage: {
        output_dir: outputDir,
        scalars_file: path.join(outputDir, 'scalars.json'),
        hdf5_file: path.join(outputDir, 'field_data.h5'),
        visualizations_dir: path.join(outputDir, 'visualizations')
      },
      progress: {
        current: 0,
        total: tier1_samples + tier2_samples + tier3_samples,
        percentage: 0
      },
      tags,
      created_by: req.user?.username || 'anonymous'
    });
    
    await dataset.save();
    
    res.status(201).json({
      success: true,
      data: dataset,
      message: 'Dataset created. Use /start endpoint to begin generation.'
    });
  } catch (error) {
    logger.error('Error creating dataset:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * POST /api/synthetic-data/datasets/:id/start
 * Start dataset generation
 */
router.post('/datasets/:id/start', async (req, res) => {
  try {
    const dataset = await SyntheticDataset.findById(req.params.id);
    
    if (!dataset) {
      return res.status(404).json({
        success: false,
        error: 'Dataset not found'
      });
    }
    
    if (dataset.status === 'generating') {
      return res.status(400).json({
        success: false,
        error: 'Dataset generation already in progress'
      });
    }
    
    if (dataset.status === 'completed') {
      return res.status(400).json({
        success: false,
        error: 'Dataset already completed'
      });
    }
    
    // Update status
    dataset.status = 'generating';
    dataset.started_at = new Date();
    await dataset.save();
    
    // Start generation process
    startGeneration(dataset);
    
    res.json({
      success: true,
      data: dataset,
      message: 'Dataset generation started'
    });
  } catch (error) {
    logger.error('Error starting generation:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * POST /api/synthetic-data/datasets/:id/cancel
 * Cancel dataset generation
 */
router.post('/datasets/:id/cancel', async (req, res) => {
  try {
    const dataset = await SyntheticDataset.findById(req.params.id);
    
    if (!dataset) {
      return res.status(404).json({
        success: false,
        error: 'Dataset not found'
      });
    }
    
    if (dataset.status !== 'generating') {
      return res.status(400).json({
        success: false,
        error: 'Dataset is not currently generating'
      });
    }
    
    // Kill the process
    const process = activeGenerations.get(dataset._id.toString());
    if (process) {
      process.kill('SIGTERM');
      activeGenerations.delete(dataset._id.toString());
    }
    
    // Update status
    dataset.status = 'cancelled';
    await dataset.save();
    
    res.json({
      success: true,
      data: dataset,
      message: 'Dataset generation cancelled'
    });
  } catch (error) {
    logger.error('Error cancelling generation:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * DELETE /api/synthetic-data/datasets/:id
 * Delete dataset
 */
router.delete('/datasets/:id', async (req, res) => {
  try {
    const dataset = await SyntheticDataset.findById(req.params.id);
    
    if (!dataset) {
      return res.status(404).json({
        success: false,
        error: 'Dataset not found'
      });
    }
    
    // Cancel if generating
    if (dataset.status === 'generating') {
      const process = activeGenerations.get(dataset._id.toString());
      if (process) {
        process.kill('SIGTERM');
        activeGenerations.delete(dataset._id.toString());
      }
    }
    
    // Delete files
    if (dataset.storage.output_dir) {
      try {
        await fs.rm(dataset.storage.output_dir, { recursive: true, force: true });
      } catch (err) {
        logger.warn('Error deleting dataset files:', err);
      }
    }
    
    // Delete samples
    await AeroSample.deleteMany({ dataset_id: dataset._id });
    
    // Delete dataset
    await dataset.deleteOne();
    
    res.json({
      success: true,
      message: 'Dataset deleted successfully'
    });
  } catch (error) {
    logger.error('Error deleting dataset:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * GET /api/synthetic-data/datasets/:id/samples
 * Get samples from a dataset
 */
router.get('/datasets/:id/samples', async (req, res) => {
  try {
    const { limit = 100, skip = 0, tier } = req.query;
    
    const query = { dataset_id: req.params.id };
    if (tier) {
      query.fidelity_tier = parseInt(tier);
    }
    
    const samples = await AeroSample.find(query)
      .sort({ created_at: -1 })
      .limit(parseInt(limit))
      .skip(parseInt(skip))
      .select('-field_data_path');
    
    const total = await AeroSample.countDocuments(query);
    
    res.json({
      success: true,
      data: samples,
      pagination: {
        total,
        limit: parseInt(limit),
        skip: parseInt(skip),
        hasMore: total > (parseInt(skip) + parseInt(limit))
      }
    });
  } catch (error) {
    logger.error('Error fetching samples:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * GET /api/synthetic-data/datasets/:id/statistics
 * Get dataset statistics
 */
router.get('/datasets/:id/statistics', async (req, res) => {
  try {
    const dataset = await SyntheticDataset.findById(req.params.id);
    
    if (!dataset) {
      return res.status(404).json({
        success: false,
        error: 'Dataset not found'
      });
    }
    
    // If completed, return stored statistics
    if (dataset.status === 'completed' && dataset.statistics.n_samples > 0) {
      return res.json({
        success: true,
        data: dataset.statistics
      });
    }
    
    // Otherwise, compute from samples
    const samples = await AeroSample.find({ dataset_id: dataset._id });
    
    if (samples.length === 0) {
      return res.json({
        success: true,
        data: { n_samples: 0 }
      });
    }
    
    const CL_values = samples.map(s => s.global_outputs.CL);
    const CD_values = samples.map(s => s.global_outputs.CD_total);
    const LD_values = samples.map(s => s.global_outputs.L_over_D);
    const balance_values = samples.map(s => s.global_outputs.balance);
    
    const statistics = {
      n_samples: samples.length,
      CL: computeStats(CL_values),
      CD: computeStats(CD_values),
      L_over_D: computeStats(LD_values),
      balance: computeStats(balance_values)
    };
    
    res.json({
      success: true,
      data: statistics
    });
  } catch (error) {
    logger.error('Error computing statistics:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * Helper function to start generation process
 */
function startGeneration(dataset) {
  const scriptPath = path.join(__dirname, '../../../..', 'synthetic_data_generation', 'batch_orchestrator.py');
  
  const args = [
    scriptPath,
    '--tier1', dataset.config.tier1_samples.toString(),
    '--tier2', dataset.config.tier2_samples.toString(),
    '--tier3', dataset.config.tier3_samples.toString(),
    '--workers', dataset.config.workers.toString(),
    '--output', dataset.storage.output_dir
  ];
  
  logger.info(`Starting generation for dataset ${dataset._id}:`, args.join(' '));
  
  const process = spawn('python', args, {
    cwd: path.join(__dirname, '../../../..', 'synthetic_data_generation')
  });
  
  activeGenerations.set(dataset._id.toString(), process);
  
  let outputBuffer = '';
  
  process.stdout.on('data', async (data) => {
    const output = data.toString();
    outputBuffer += output;
    logger.info(`[Dataset ${dataset._id}] ${output.trim()}`);
    
    // Parse progress from output
    const progressMatch = output.match(/(\d+)\/(\d+)/);
    if (progressMatch) {
      const current = parseInt(progressMatch[1]);
      const total = parseInt(progressMatch[2]);
      
      try {
        await dataset.updateProgress(current, total);
      } catch (err) {
        logger.error('Error updating progress:', err);
      }
    }
  });
  
  process.stderr.on('data', (data) => {
    logger.error(`[Dataset ${dataset._id}] Error: ${data.toString()}`);
  });
  
  process.on('close', async (code) => {
    activeGenerations.delete(dataset._id.toString());
    
    try {
      if (code === 0) {
        // Success - load statistics
        const scalarsPath = dataset.storage.scalars_file;
        
        try {
          const scalarsData = await fs.readFile(scalarsPath, 'utf8');
          const samples = JSON.parse(scalarsData);
          
          // Compute statistics
          const CL_values = samples.map(s => s.global_outputs.CL);
          const CD_values = samples.map(s => s.global_outputs.CD_total);
          const LD_values = samples.map(s => s.global_outputs.L_over_D);
          const balance_values = samples.map(s => s.global_outputs.balance);
          
          const statistics = {
            n_samples: samples.length,
            CL: computeStats(CL_values),
            CD: computeStats(CD_values),
            L_over_D: computeStats(LD_values),
            balance: computeStats(balance_values)
          };
          
          // Save samples to MongoDB (in batches)
          const batchSize = 100;
          for (let i = 0; i < samples.length; i += batchSize) {
            const batch = samples.slice(i, i + batchSize).map(s => ({
              sample_id: s.sample_id,
              dataset_id: dataset._id,
              timestamp: new Date(s.timestamp),
              fidelity_tier: s.fidelity_tier,
              geometry_params: s.geometry_params,
              flow_conditions: s.flow_conditions,
              global_outputs: s.global_outputs,
              provenance: s.provenance
            }));
            
            await AeroSample.insertMany(batch, { ordered: false });
          }
          
          await dataset.markCompleted(statistics);
          logger.info(`Dataset ${dataset._id} completed successfully`);
        } catch (err) {
          logger.error('Error processing results:', err);
          await dataset.markFailed(err);
        }
      } else {
        // Failed
        const error = new Error(`Generation process exited with code ${code}`);
        await dataset.markFailed(error);
        logger.error(`Dataset ${dataset._id} failed with code ${code}`);
      }
    } catch (err) {
      logger.error('Error in process close handler:', err);
    }
  });
}

/**
 * Helper function to compute statistics
 */
function computeStats(values) {
  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
  const std = Math.sqrt(variance);
  const min = Math.min(...values);
  const max = Math.max(...values);
  
  return { mean, std, min, max };
}

module.exports = router;
