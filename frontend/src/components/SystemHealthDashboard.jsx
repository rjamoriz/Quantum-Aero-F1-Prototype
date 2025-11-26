/**
 * System Health Dashboard
 * Monitor all microservices, resources, and system metrics
 */

import React, { useState, useEffect } from 'react';
import { Activity, Cpu, HardDrive, Zap, Server, Database, AlertCircle, CheckCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const SystemHealthDashboard = () => {
  const [services, setServices] = useState([]);
  const [metrics, setMetrics] = useState({
    cpu: [],
    memory: [],
    gpu: []
  });

  useEffect(() => {
    loadServiceStatus();
    const interval = setInterval(loadServiceStatus, 5000); // Poll every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const loadServiceStatus = async () => {
    // Mock service status (would call actual health endpoints)
    const mockServices = [
      {
        name: 'ML Inference Service',
        endpoint: 'http://localhost:8000',
        status: 'healthy',
        uptime: '99.8%',
        latency: 87,
        requests: 1543,
        errors: 3,
        version: 'v2.1.0'
      },
      {
        name: 'Physics Engine',
        endpoint: 'http://localhost:8001',
        status: 'healthy',
        uptime: '99.9%',
        latency: 245,
        requests: 892,
        errors: 1,
        version: 'v1.5.2'
      },
      {
        name: 'Quantum Optimizer',
        endpoint: 'http://localhost:8002',
        status: 'healthy',
        uptime: '98.5%',
        latency: 8300,
        requests: 234,
        errors: 5,
        version: 'v1.2.1'
      },
      {
        name: 'Backend API',
        endpoint: 'http://localhost:3001',
        status: 'healthy',
        uptime: '99.7%',
        latency: 45,
        requests: 3421,
        errors: 8,
        version: 'v3.0.0'
      },
      {
        name: 'MongoDB',
        endpoint: 'mongodb://localhost:27017',
        status: 'healthy',
        uptime: '100%',
        latency: 12,
        requests: 5234,
        errors: 0,
        version: '7.0.4'
      }
    ];

    setServices(mockServices);

    // Generate mock metrics
    const now = Date.now();
    const newMetrics = {
      cpu: [...metrics.cpu, { time: now, value: 45 + Math.random() * 20 }].slice(-20),
      memory: [...metrics.memory, { time: now, value: 60 + Math.random() * 15 }].slice(-20),
      gpu: [...metrics.gpu, { time: now, value: 70 + Math.random() * 25 }].slice(-20)
    };
    setMetrics(newMetrics);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-100 border-green-300 text-green-800';
      case 'degraded':
        return 'bg-yellow-100 border-yellow-300 text-yellow-800';
      case 'down':
        return 'bg-red-100 border-red-300 text-red-800';
      default:
        return 'bg-gray-100 border-gray-300 text-gray-800';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'degraded':
        return <AlertCircle className="w-5 h-5 text-yellow-600" />;
      case 'down':
        return <AlertCircle className="w-5 h-5 text-red-600" />;
      default:
        return <Activity className="w-5 h-5 text-gray-600" />;
    }
  };

  const healthyServices = services.filter(s => s.status === 'healthy').length;
  const totalServices = services.length;
  const avgLatency = services.reduce((sum, s) => sum + s.latency, 0) / totalServices || 0;
  const totalRequests = services.reduce((sum, s) => sum + s.requests, 0);
  const totalErrors = services.reduce((sum, s) => sum + s.errors, 0);
  const errorRate = totalRequests > 0 ? (totalErrors / totalRequests * 100) : 0;

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
        <Activity className="w-6 h-6" />
        System Health Dashboard
      </h2>

      <p className="text-gray-600 mb-6">
        Real-time monitoring of all microservices, resources, and system performance metrics.
      </p>

      {/* Overall Status */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="p-4 bg-green-50 border border-green-200 rounded">
          <div className="flex items-center gap-2 mb-2">
            <Server className="w-5 h-5 text-green-600" />
            <span className="text-sm text-green-700">Services Online</span>
          </div>
          <div className="text-3xl font-bold text-green-900">
            {healthyServices}/{totalServices}
          </div>
        </div>

        <div className="p-4 bg-blue-50 border border-blue-200 rounded">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-5 h-5 text-blue-600" />
            <span className="text-sm text-blue-700">Avg Latency</span>
          </div>
          <div className="text-3xl font-bold text-blue-900">
            {avgLatency.toFixed(0)}ms
          </div>
        </div>

        <div className="p-4 bg-purple-50 border border-purple-200 rounded">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-5 h-5 text-purple-600" />
            <span className="text-sm text-purple-700">Total Requests</span>
          </div>
          <div className="text-3xl font-bold text-purple-900">
            {totalRequests.toLocaleString()}
          </div>
        </div>

        <div className="p-4 bg-red-50 border border-red-200 rounded">
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle className="w-5 h-5 text-red-600" />
            <span className="text-sm text-red-700">Error Rate</span>
          </div>
          <div className="text-3xl font-bold text-red-900">
            {errorRate.toFixed(2)}%
          </div>
        </div>
      </div>

      {/* Service Status */}
      <div className="mb-6">
        <h3 className="font-semibold mb-3">Service Status</h3>
        <div className="space-y-3">
          {services.map((service, idx) => (
            <div
              key={idx}
              className={`p-4 rounded border-2 ${getStatusColor(service.status)}`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {getStatusIcon(service.status)}
                  <div>
                    <div className="font-semibold">{service.name}</div>
                    <div className="text-xs opacity-75">{service.endpoint}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs opacity-75">Version</div>
                  <div className="font-mono text-sm">{service.version}</div>
                </div>
              </div>

              <div className="grid grid-cols-5 gap-4 mt-3 text-sm">
                <div>
                  <div className="text-xs opacity-75">Status</div>
                  <div className="font-semibold capitalize">{service.status}</div>
                </div>
                <div>
                  <div className="text-xs opacity-75">Uptime</div>
                  <div className="font-semibold">{service.uptime}</div>
                </div>
                <div>
                  <div className="text-xs opacity-75">Latency</div>
                  <div className="font-semibold">{service.latency}ms</div>
                </div>
                <div>
                  <div className="text-xs opacity-75">Requests</div>
                  <div className="font-semibold">{service.requests.toLocaleString()}</div>
                </div>
                <div>
                  <div className="text-xs opacity-75">Errors</div>
                  <div className={`font-semibold ${service.errors > 0 ? 'text-red-600' : ''}`}>
                    {service.errors}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Resource Metrics */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="p-4 bg-gray-50 rounded border border-gray-200">
          <div className="flex items-center gap-2 mb-3">
            <Cpu className="w-5 h-5 text-blue-600" />
            <h4 className="font-semibold">CPU Usage</h4>
          </div>
          <ResponsiveContainer width="100%" height={120}>
            <LineChart data={metrics.cpu}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" hide />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <div className="text-center mt-2">
            <span className="text-2xl font-bold text-blue-600">
              {metrics.cpu.length > 0 ? metrics.cpu[metrics.cpu.length - 1].value.toFixed(1) : 0}%
            </span>
          </div>
        </div>

        <div className="p-4 bg-gray-50 rounded border border-gray-200">
          <div className="flex items-center gap-2 mb-3">
            <HardDrive className="w-5 h-5 text-green-600" />
            <h4 className="font-semibold">Memory Usage</h4>
          </div>
          <ResponsiveContainer width="100%" height={120}>
            <LineChart data={metrics.memory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" hide />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Line type="monotone" dataKey="value" stroke="#10b981" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <div className="text-center mt-2">
            <span className="text-2xl font-bold text-green-600">
              {metrics.memory.length > 0 ? metrics.memory[metrics.memory.length - 1].value.toFixed(1) : 0}%
            </span>
          </div>
        </div>

        <div className="p-4 bg-gray-50 rounded border border-gray-200">
          <div className="flex items-center gap-2 mb-3">
            <Zap className="w-5 h-5 text-purple-600" />
            <h4 className="font-semibold">GPU Usage</h4>
          </div>
          <ResponsiveContainer width="100%" height={120}>
            <LineChart data={metrics.gpu}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" hide />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Line type="monotone" dataKey="value" stroke="#8b5cf6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <div className="text-center mt-2">
            <span className="text-2xl font-bold text-purple-600">
              {metrics.gpu.length > 0 ? metrics.gpu[metrics.gpu.length - 1].value.toFixed(1) : 0}%
            </span>
          </div>
        </div>
      </div>

      {/* Alerts */}
      <div className="p-4 bg-yellow-50 border border-yellow-200 rounded">
        <h3 className="font-semibold mb-3 flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-yellow-600" />
          System Alerts
        </h3>
        <div className="space-y-2 text-sm">
          {errorRate > 1 && (
            <div className="flex items-start gap-2">
              <span className="text-red-600">⚠️</span>
              <p>Error rate above threshold ({errorRate.toFixed(2)}%). Investigate failed requests.</p>
            </div>
          )}
          {avgLatency > 1000 && (
            <div className="flex items-start gap-2">
              <span className="text-yellow-600">⚠️</span>
              <p>Average latency high ({avgLatency.toFixed(0)}ms). Consider scaling services.</p>
            </div>
          )}
          {metrics.gpu.length > 0 && metrics.gpu[metrics.gpu.length - 1].value > 90 && (
            <div className="flex items-start gap-2">
              <span className="text-orange-600">⚠️</span>
              <p>GPU usage critical (&gt;90%). ML inference may be throttled.</p>
            </div>
          )}
          {healthyServices === totalServices && errorRate < 1 && avgLatency < 1000 && (
            <div className="flex items-start gap-2">
              <span className="text-green-600">✓</span>
              <p>All systems operational. No alerts at this time.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SystemHealthDashboard;
