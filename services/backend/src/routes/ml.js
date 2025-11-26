/**
 * ML Surrogate Routes
 * Proxy routes to the ML Surrogate microservice (GPU inference)
 */

const express = require('express');
const router = express.Router();
const { createServiceClient, cachedRequest, healthCheck } = require('../utils/serviceClient');
const { ml: mlConfig } = require('../config/services');
const logger = require('../utils/logger');

// Create ML service client
const mlClient = createServiceClient(
  'ML Surrogate',
  mlConfig.baseUrl,
  mlConfig.timeout
);

/**
 * POST /api/ml/predict
 * Predict aerodynamic quantities using ML surrogate
 */
router.post('/predict', async (req, res, next) => {
  try {
    logger.info('ML prediction request received');

    const { mesh_id, parameters, use_cache = true } = req.body;

    if (use_cache) {
      // Try cache first
      const cacheKey = `ml:predict:${mesh_id}:${JSON.stringify(parameters)}`;
      const result = await cachedRequest(
        mlClient,
        {
          method: 'POST',
          url: mlConfig.endpoints.predict,
          data: req.body,
        },
        cacheKey,
        3600 // 1 hour cache
      );

      return res.json({
        success: true,
        data: result.data,
        cached: result.cached,
        service: 'ml-surrogate',
        timestamp: new Date().toISOString(),
      });
    }

    // No cache - direct request
    const response = await mlClient.post(mlConfig.endpoints.predict, req.body);

    res.json({
      success: true,
      data: response.data,
      cached: false,
      service: 'ml-surrogate',
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    next(error);
  }
});

/**
 * POST /api/ml/predict/batch
 * Batch prediction for multiple designs
 */
router.post('/predict/batch', async (req, res, next) => {
  try {
    logger.info('ML batch prediction request received');

    const response = await mlClient.post(
      mlConfig.endpoints.predictBatch,
      req.body
    );

    res.json({
      success: true,
      data: response.data,
      service: 'ml-surrogate',
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    next(error);
  }
});

/**
 * GET /api/ml/models
 * Get available ML models
 */
router.get('/models', async (req, res, next) => {
  try {
    const response = await mlClient.get(mlConfig.endpoints.models);

    res.json({
      success: true,
      data: response.data,
      service: 'ml-surrogate',
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    next(error);
  }
});

/**
 * GET /api/ml/health
 * Check ML service health
 */
router.get('/health', async (req, res) => {
  const health = await healthCheck(mlClient, mlConfig.endpoints.health);
  
  res.status(health.healthy ? 200 : 503).json({
    service: 'ml-surrogate',
    ...health,
    timestamp: new Date().toISOString(),
  });
});

module.exports = router;
