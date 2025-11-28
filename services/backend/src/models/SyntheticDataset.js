/**
 * MongoDB model for Synthetic Aerodynamic Datasets
 */

const mongoose = require('mongoose');

const GeometryParametersSchema = new mongoose.Schema({
  main_plane_chord: { type: Number, required: true },
  main_plane_span: { type: Number, required: true },
  main_plane_angle_deg: { type: Number, required: true },
  flap1_angle_deg: { type: Number, required: true },
  flap2_angle_deg: { type: Number, required: true },
  endplate_height: { type: Number, required: true },
  rear_wing_chord: { type: Number, required: true },
  rear_wing_span: { type: Number, required: true },
  rear_wing_angle_deg: { type: Number, required: true },
  beam_wing_angle: { type: Number, required: true },
  floor_gap: { type: Number, required: true },
  diffuser_angle: { type: Number, required: true },
  diffuser_length: { type: Number, required: true },
  sidepod_width: { type: Number, required: true },
  sidepod_undercut: { type: Number, required: true },
  DRS_open: { type: Boolean, required: true }
}, { _id: false });

const FlowConditionsSchema = new mongoose.Schema({
  V_inf: { type: Number, required: true },
  rho: { type: Number, required: true },
  nu: { type: Number, required: true },
  yaw: { type: Number, default: 0 },
  roll: { type: Number, default: 0 },
  pitch: { type: Number, default: 0 },
  ground_gap: { type: Number, required: true },
  Re: { type: Number, required: true }
}, { _id: false });

const GlobalOutputsSchema = new mongoose.Schema({
  CL: { type: Number, required: true },
  CD_total: { type: Number, required: true },
  CD_induced: { type: Number, required: true },
  CD_pressure: Number,
  CD_friction: Number,
  CM: { type: Number, default: 0 },
  L: { type: Number, default: 0 },
  D: { type: Number, default: 0 },
  downforce_front: { type: Number, default: 0 },
  downforce_rear: { type: Number, default: 0 },
  balance: { type: Number, default: 0 },
  L_over_D: { type: Number, default: 0 },
  flutter_risk_estimate: Number
}, { _id: false });

const ProvenanceSchema = new mongoose.Schema({
  solver: { type: String, required: true },
  mesh_size: Number,
  residuals: Number,
  estimated_error: Number,
  runtime_seconds: Number,
  convergence_achieved: { type: Boolean, default: true },
  notes: String
}, { _id: false });

// Individual sample schema
const AeroSampleSchema = new mongoose.Schema({
  sample_id: { type: String, required: true, unique: true, index: true },
  dataset_id: { type: mongoose.Schema.Types.ObjectId, ref: 'SyntheticDataset', required: true, index: true },
  timestamp: { type: Date, default: Date.now },
  fidelity_tier: { type: Number, required: true, min: 0, max: 4 },
  geometry_params: { type: GeometryParametersSchema, required: true },
  flow_conditions: { type: FlowConditionsSchema, required: true },
  global_outputs: { type: GlobalOutputsSchema, required: true },
  provenance: { type: ProvenanceSchema, required: true },
  field_data_path: String, // Path to HDF5 file for field data
  created_at: { type: Date, default: Date.now }
});

// Main dataset schema
const SyntheticDatasetSchema = new mongoose.Schema({
  name: { type: String, required: true },
  description: String,
  
  // Generation configuration
  config: {
    tier1_samples: { type: Number, default: 0 },
    tier2_samples: { type: Number, default: 0 },
    tier3_samples: { type: Number, default: 0 },
    workers: { type: Number, default: 4 },
    sampling_method: { type: String, enum: ['lhs', 'stratified'], default: 'lhs' }
  },
  
  // Generation status
  status: {
    type: String,
    enum: ['pending', 'generating', 'completed', 'failed', 'cancelled'],
    default: 'pending',
    index: true
  },
  
  progress: {
    current: { type: Number, default: 0 },
    total: { type: Number, default: 0 },
    percentage: { type: Number, default: 0 },
    estimated_time_remaining: Number, // seconds
    samples_per_second: Number
  },
  
  // Statistics (computed after generation)
  statistics: {
    n_samples: { type: Number, default: 0 },
    CL: {
      mean: Number,
      std: Number,
      min: Number,
      max: Number
    },
    CD: {
      mean: Number,
      std: Number,
      min: Number,
      max: Number
    },
    L_over_D: {
      mean: Number,
      std: Number,
      min: Number,
      max: Number
    },
    balance: {
      mean: Number,
      std: Number,
      min: Number,
      max: Number
    }
  },
  
  // Storage paths
  storage: {
    output_dir: String,
    scalars_file: String,
    hdf5_file: String,
    visualizations_dir: String,
    total_size_mb: Number
  },
  
  // Metadata
  created_by: { type: String, default: 'system' },
  created_at: { type: Date, default: Date.now },
  started_at: Date,
  completed_at: Date,
  
  // Error tracking
  error: {
    message: String,
    stack: String,
    timestamp: Date
  },
  
  // Tags for organization
  tags: [String]
});

// Indexes for efficient queries
SyntheticDatasetSchema.index({ status: 1, created_at: -1 });
SyntheticDatasetSchema.index({ 'statistics.n_samples': -1 });
SyntheticDatasetSchema.index({ tags: 1 });

AeroSampleSchema.index({ dataset_id: 1, fidelity_tier: 1 });
AeroSampleSchema.index({ 'global_outputs.CL': 1 });
AeroSampleSchema.index({ 'global_outputs.L_over_D': -1 });

// Virtual for total samples
SyntheticDatasetSchema.virtual('total_samples').get(function() {
  return this.config.tier1_samples + this.config.tier2_samples + this.config.tier3_samples;
});

// Method to update progress
SyntheticDatasetSchema.methods.updateProgress = function(current, total) {
  this.progress.current = current;
  this.progress.total = total;
  this.progress.percentage = Math.round((current / total) * 100);
  
  // Calculate samples per second
  if (this.started_at) {
    const elapsed = (Date.now() - this.started_at) / 1000; // seconds
    this.progress.samples_per_second = current / elapsed;
    
    // Estimate time remaining
    const remaining_samples = total - current;
    this.progress.estimated_time_remaining = remaining_samples / this.progress.samples_per_second;
  }
  
  return this.save();
};

// Method to mark as completed
SyntheticDatasetSchema.methods.markCompleted = function(statistics) {
  this.status = 'completed';
  this.completed_at = new Date();
  this.progress.percentage = 100;
  
  if (statistics) {
    this.statistics = statistics;
  }
  
  return this.save();
};

// Method to mark as failed
SyntheticDatasetSchema.methods.markFailed = function(error) {
  this.status = 'failed';
  this.error = {
    message: error.message,
    stack: error.stack,
    timestamp: new Date()
  };
  
  return this.save();
};

// Static method to get recent datasets
SyntheticDatasetSchema.statics.getRecent = function(limit = 10) {
  return this.find()
    .sort({ created_at: -1 })
    .limit(limit)
    .select('-error.stack');
};

// Static method to get active generations
SyntheticDatasetSchema.statics.getActive = function() {
  return this.find({ status: 'generating' })
    .sort({ started_at: -1 });
};

const SyntheticDataset = mongoose.model('SyntheticDataset', SyntheticDatasetSchema);
const AeroSample = mongoose.model('AeroSample', AeroSampleSchema);

module.exports = {
  SyntheticDataset,
  AeroSample
};
