/**
 * Service Client Utility
 * HTTP client for communicating with microservices
 */

const axios = require('axios');
const logger = require('./logger');
const { cache } = require('../config/redis');

/**
 * Create HTTP client for a microservice
 */
function createServiceClient(serviceName, baseUrl, timeout = 30000) {
  const client = axios.create({
    baseURL: baseUrl,
    timeout: timeout,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Request interceptor
  client.interceptors.request.use(
    (config) => {
      logger.debug(`${serviceName} request: ${config.method.toUpperCase()} ${config.url}`);
      return config;
    },
    (error) => {
      logger.error(`${serviceName} request error:`, error.message);
      return Promise.reject(error);
    }
  );

  // Response interceptor
  client.interceptors.response.use(
    (response) => {
      logger.debug(`${serviceName} response: ${response.status} ${response.config.url}`);
      return response;
    },
    (error) => {
      if (error.response) {
        // Server responded with error status
        logger.error(
          `${serviceName} error: ${error.response.status} ${error.response.config.url}`,
          error.response.data
        );
      } else if (error.request) {
        // Request made but no response
        logger.error(`${serviceName} no response:`, error.message);
      } else {
        // Error setting up request
        logger.error(`${serviceName} request setup error:`, error.message);
      }
      return Promise.reject(error);
    }
  );

  return client;
}

/**
 * Make cached request to service
 */
async function cachedRequest(client, config, cacheKey, cacheTTL = 3600) {
  try {
    // Check cache first
    const cached = await cache.get(cacheKey);
    if (cached) {
      logger.debug(`Cache hit: ${cacheKey}`);
      return { data: cached, cached: true };
    }

    // Make request
    const response = await client(config);

    // Cache response
    await cache.set(cacheKey, response.data, cacheTTL);
    logger.debug(`Cache set: ${cacheKey}`);

    return { data: response.data, cached: false };
  } catch (error) {
    throw error;
  }
}

/**
 * Health check for a service
 */
async function healthCheck(client, healthEndpoint = '/health') {
  try {
    const response = await client.get(healthEndpoint, { timeout: 5000 });
    return {
      healthy: response.status === 200,
      status: response.data,
    };
  } catch (error) {
    return {
      healthy: false,
      error: error.message,
    };
  }
}

/**
 * Retry logic for failed requests
 */
async function retryRequest(fn, maxRetries = 3, delay = 1000) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      
      logger.warn(`Request failed, retrying (${i + 1}/${maxRetries})...`);
      await new Promise(resolve => setTimeout(resolve, delay * (i + 1)));
    }
  }
}

module.exports = {
  createServiceClient,
  cachedRequest,
  healthCheck,
  retryRequest,
};
