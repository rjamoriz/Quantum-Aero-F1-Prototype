/**
 * Main Express server with synthetic data routes
 */

const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const syntheticDataRoutes = require('./routes/synthetic-data');
const quantumOptimizationRoutes = require('./routes/quantum-optimization');

const app = express();
const PORT = process.env.PORT || 8000;

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB connection
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/f1-quantum-aero';

mongoose.connect(MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
.then(() => console.log('✓ MongoDB connected'))
.catch(err => console.error('MongoDB connection error:', err));

// Routes
app.use('/api/synthetic-data', syntheticDataRoutes);
app.use('/api/quantum', quantumOptimizationRoutes);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`📊 Synthetic Data API: http://localhost:${PORT}/api/synthetic-data`);
  console.log(`⚛️  Quantum Optimization API: http://localhost:${PORT}/api/quantum`);
});

module.exports = app;
