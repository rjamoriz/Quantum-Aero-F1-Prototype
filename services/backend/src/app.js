/**
 * Quantum-Aero F1 Backend API Gateway
 * 
 * Main Express application that orchestrates:
 * - Physics Engine (VLM/Panel)
 * - ML Surrogate (GPU inference)
 * - Quantum Optimizer (QAOA/QUBO)
 * - GenAI Agents (Claude)
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const compression = require('compression');
require('dotenv').config();

const logger = require('./utils/logger');
const { connectDB } = require('./config/database');
const { connectRedis } = require('./config/redis');

// Import routes
const physicsRoutes = require('./routes/physics');
const mlRoutes = require('./routes/ml');
const quantumRoutes = require('./routes/quantum');
const claudeRoutes = require('./routes/claude');
const simulationRoutes = require('./routes/simulation');

// Initialize Express app
const app = express();

// Middleware
app.use(helmet()); // Security headers
app.use(cors()); // CORS
app.use(compression()); // Response compression
app.use(express.json({ limit: '10mb' })); // JSON parser
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(morgan('combined', { stream: { write: message => logger.info(message.trim()) } }));

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'quantum-aero-backend',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// API Routes
app.use('/api/physics', physicsRoutes);
app.use('/api/ml', mlRoutes);
app.use('/api/quantum', quantumRoutes);
app.use('/api/claude', claudeRoutes);
app.use('/api/simulation', simulationRoutes);

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    message: 'Quantum-Aero F1 Backend API',
    version: '1.0.0',
    endpoints: {
      health: '/health',
      physics: '/api/physics',
      ml: '/api/ml',
      quantum: '/api/quantum',
      claude: '/api/claude',
      simulation: '/api/simulation'
    }
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  logger.error('Error:', err);
  
  res.status(err.status || 500).json({
    error: {
      message: err.message || 'Internal server error',
      status: err.status || 500,
      timestamp: new Date().toISOString()
    }
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: {
      message: 'Endpoint not found',
      status: 404,
      path: req.path
    }
  });
});

// Start server
const PORT = process.env.PORT || 3001;

async function startServer() {
  try {
    // Connect to MongoDB
    await connectDB();
    logger.info('MongoDB connected');
    
    // Connect to Redis
    await connectRedis();
    logger.info('Redis connected');
    
    // Start Express server
    app.listen(PORT, () => {
      logger.info(`🚀 Backend API running on port ${PORT}`);
      logger.info(`📊 Health check: http://localhost:${PORT}/health`);
      logger.info(`📚 API docs: http://localhost:${PORT}/`);
    });
    
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  logger.info('SIGINT received, shutting down gracefully');
  process.exit(0);
});

// Start the server
startServer();

module.exports = app;
