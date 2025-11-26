/**
 * Job Model
 * MongoDB schema for simulation jobs
 */

const mongoose = require('mongoose');

const jobSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },
  
  type: {
    type: String,
    required: true,
    enum: [
      'ml_inference',
      'physics_validation',
      'quantum_optimization',
      'transient_simulation',
      'fsi_validation',
      'multi_fidelity'
    ],
    index: true
  },
  
  status: {
    type: String,
    required: true,
    enum: ['pending', 'running', 'completed', 'failed', 'cancelled'],
    default: 'pending',
    index: true
  },
  
  priority: {
    type: String,
    enum: ['low', 'medium', 'high'],
    default: 'medium',
    index: true
  },
  
  progress: {
    type: Number,
    default: 0,
    min: 0,
    max: 100
  },
  
  parameters: {
    type: mongoose.Schema.Types.Mixed,
    required: true
  },
  
  result: {
    type: mongoose.Schema.Types.Mixed,
    default: null
  },
  
  error: {
    type: String,
    default: null
  },
  
  // Timestamps
  created: {
    type: Date,
    default: Date.now,
    index: true
  },
  
  started: {
    type: Date,
    default: null
  },
  
  completed: {
    type: Date,
    default: null
  },
  
  failed: {
    type: Date,
    default: null
  },
  
  cancelled: {
    type: Date,
    default: null
  },
  
  retried: {
    type: Date,
    default: null
  },
  
  // Duration in seconds
  duration: {
    type: Number,
    default: null
  },
  
  // Retry tracking
  retryCount: {
    type: Number,
    default: 0
  },
  
  // Service metadata
  serviceEndpoint: {
    type: String,
    default: null
  },
  
  serviceVersion: {
    type: String,
    default: null
  }
}, {
  timestamps: true
});

// Indexes for efficient querying
jobSchema.index({ userId: 1, status: 1, created: -1 });
jobSchema.index({ userId: 1, type: 1, created: -1 });
jobSchema.index({ status: 1, priority: -1, created: 1 }); // For job queue processing

// Virtual for elapsed time
jobSchema.virtual('elapsedTime').get(function() {
  if (this.started) {
    const end = this.completed || this.failed || new Date();
    return (end - this.started) / 1000; // seconds
  }
  return null;
});

// Method to update progress
jobSchema.methods.updateProgress = async function(progress) {
  this.progress = Math.min(Math.max(progress, 0), 100);
  return this.save();
};

// Method to mark as completed
jobSchema.methods.markCompleted = async function(result) {
  this.status = 'completed';
  this.progress = 100;
  this.completed = new Date();
  this.result = result;
  if (this.started) {
    this.duration = (this.completed - this.started) / 1000;
  }
  return this.save();
};

// Method to mark as failed
jobSchema.methods.markFailed = async function(error) {
  this.status = 'failed';
  this.failed = new Date();
  this.error = error;
  return this.save();
};

// Static method to get job statistics
jobSchema.statics.getStatistics = async function(userId) {
  const stats = await this.aggregate([
    { $match: { userId: mongoose.Types.ObjectId(userId) } },
    {
      $group: {
        _id: '$status',
        count: { $sum: 1 }
      }
    }
  ]);
  
  const result = {
    total: 0,
    pending: 0,
    running: 0,
    completed: 0,
    failed: 0,
    cancelled: 0
  };
  
  stats.forEach(stat => {
    result[stat._id] = stat.count;
    result.total += stat.count;
  });
  
  return result;
};

// Static method to cleanup old jobs
jobSchema.statics.cleanupOldJobs = async function(daysOld = 30) {
  const cutoffDate = new Date();
  cutoffDate.setDate(cutoffDate.getDate() - daysOld);
  
  const result = await this.deleteMany({
    status: { $in: ['completed', 'failed', 'cancelled'] },
    created: { $lt: cutoffDate }
  });
  
  return result.deletedCount;
};

const Job = mongoose.model('Job', jobSchema);

module.exports = Job;
