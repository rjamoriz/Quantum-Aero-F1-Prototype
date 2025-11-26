/**
 * Job Orchestration Dashboard
 * Track and manage simulation jobs across all services
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Play, Pause, RotateCcw, X, Clock, CheckCircle, XCircle, AlertCircle } from 'lucide-react';

const JobOrchestrationDashboard = () => {
  const [jobs, setJobs] = useState([]);
  const [filter, setFilter] = useState('all'); // all, pending, running, completed, failed
  const [sortBy, setSortBy] = useState('created'); // created, priority, status

  useEffect(() => {
    loadJobs();
    const interval = setInterval(loadJobs, 2000); // Poll every 2 seconds
    return () => clearInterval(interval);
  }, [filter, sortBy]);

  const loadJobs = async () => {
    try {
      const response = await axios.get(`http://localhost:3001/api/jobs`, {
        params: { filter, sortBy }
      });
      setJobs(response.data.jobs);
    } catch (error) {
      // Use mock data if API not available
      setJobs(getMockJobs());
    }
  };

  const getMockJobs = () => [
    {
      id: 'job_001',
      type: 'ml_inference',
      status: 'completed',
      priority: 'high',
      progress: 100,
      created: new Date(Date.now() - 300000).toISOString(),
      started: new Date(Date.now() - 240000).toISOString(),
      completed: new Date(Date.now() - 180000).toISOString(),
      duration: 60,
      parameters: { mesh_id: 'wing_v3.2', velocity: 250 },
      result: { Cl: 2.8, Cd: 0.42 }
    },
    {
      id: 'job_002',
      type: 'quantum_optimization',
      status: 'running',
      priority: 'high',
      progress: 65,
      created: new Date(Date.now() - 120000).toISOString(),
      started: new Date(Date.now() - 90000).toISOString(),
      parameters: { n_variables: 20, method: 'QAOA' }
    },
    {
      id: 'job_003',
      type: 'transient_simulation',
      status: 'pending',
      priority: 'medium',
      progress: 0,
      created: new Date(Date.now() - 60000).toISOString(),
      parameters: { scenario: 'corner_exit' }
    },
    {
      id: 'job_004',
      type: 'fsi_validation',
      status: 'failed',
      priority: 'low',
      progress: 45,
      created: new Date(Date.now() - 180000).toISOString(),
      started: new Date(Date.now() - 150000).toISOString(),
      failed: new Date(Date.now() - 120000).toISOString(),
      error: 'Solver divergence at t=2.5s'
    }
  ];

  const retryJob = async (jobId) => {
    try {
      await axios.put(`http://localhost:3001/api/jobs/${jobId}/retry`);
      loadJobs();
    } catch (error) {
      console.error('Retry failed:', error);
    }
  };

  const cancelJob = async (jobId) => {
    try {
      await axios.delete(`http://localhost:3001/api/jobs/${jobId}`);
      loadJobs();
    } catch (error) {
      console.error('Cancel failed:', error);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'running':
        return <Clock className="w-5 h-5 text-blue-600 animate-spin" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-600" />;
      case 'pending':
        return <AlertCircle className="w-5 h-5 text-yellow-600" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'running':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'failed':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'pending':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high':
        return 'text-red-600 font-bold';
      case 'medium':
        return 'text-yellow-600 font-semibold';
      case 'low':
        return 'text-gray-600';
      default:
        return 'text-gray-600';
    }
  };

  const formatDuration = (seconds) => {
    if (!seconds) return '-';
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  };

  const formatTime = (isoString) => {
    if (!isoString) return '-';
    return new Date(isoString).toLocaleTimeString();
  };

  const filteredJobs = jobs.filter(job => {
    if (filter === 'all') return true;
    return job.status === filter;
  });

  const stats = {
    total: jobs.length,
    pending: jobs.filter(j => j.status === 'pending').length,
    running: jobs.filter(j => j.status === 'running').length,
    completed: jobs.filter(j => j.status === 'completed').length,
    failed: jobs.filter(j => j.status === 'failed').length
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4">Job Orchestration Dashboard</h2>

      {/* Statistics */}
      <div className="grid grid-cols-5 gap-4 mb-6">
        <div className="p-4 bg-gray-50 rounded border border-gray-200">
          <div className="text-sm text-gray-600">Total Jobs</div>
          <div className="text-2xl font-bold">{stats.total}</div>
        </div>
        <div className="p-4 bg-yellow-50 rounded border border-yellow-200">
          <div className="text-sm text-yellow-700">Pending</div>
          <div className="text-2xl font-bold text-yellow-800">{stats.pending}</div>
        </div>
        <div className="p-4 bg-blue-50 rounded border border-blue-200">
          <div className="text-sm text-blue-700">Running</div>
          <div className="text-2xl font-bold text-blue-800">{stats.running}</div>
        </div>
        <div className="p-4 bg-green-50 rounded border border-green-200">
          <div className="text-sm text-green-700">Completed</div>
          <div className="text-2xl font-bold text-green-800">{stats.completed}</div>
        </div>
        <div className="p-4 bg-red-50 rounded border border-red-200">
          <div className="text-sm text-red-700">Failed</div>
          <div className="text-2xl font-bold text-red-800">{stats.failed}</div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex justify-between items-center mb-4">
        <div className="flex gap-2">
          <button
            onClick={() => setFilter('all')}
            className={`px-4 py-2 rounded ${filter === 'all' ? 'bg-purple-600 text-white' : 'bg-gray-200'}`}
          >
            All
          </button>
          <button
            onClick={() => setFilter('pending')}
            className={`px-4 py-2 rounded ${filter === 'pending' ? 'bg-yellow-600 text-white' : 'bg-gray-200'}`}
          >
            Pending
          </button>
          <button
            onClick={() => setFilter('running')}
            className={`px-4 py-2 rounded ${filter === 'running' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
          >
            Running
          </button>
          <button
            onClick={() => setFilter('completed')}
            className={`px-4 py-2 rounded ${filter === 'completed' ? 'bg-green-600 text-white' : 'bg-gray-200'}`}
          >
            Completed
          </button>
          <button
            onClick={() => setFilter('failed')}
            className={`px-4 py-2 rounded ${filter === 'failed' ? 'bg-red-600 text-white' : 'bg-gray-200'}`}
          >
            Failed
          </button>
        </div>

        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
          className="px-4 py-2 border rounded"
        >
          <option value="created">Sort by Created</option>
          <option value="priority">Sort by Priority</option>
          <option value="status">Sort by Status</option>
        </select>
      </div>

      {/* Job List */}
      <div className="space-y-3">
        {filteredJobs.map(job => (
          <div
            key={job.id}
            className={`p-4 rounded border ${getStatusColor(job.status)}`}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  {getStatusIcon(job.status)}
                  <span className="font-semibold text-lg">{job.id}</span>
                  <span className="px-2 py-1 bg-white rounded text-sm">{job.type}</span>
                  <span className={`text-sm ${getPriorityColor(job.priority)}`}>
                    {job.priority.toUpperCase()} PRIORITY
                  </span>
                </div>

                {/* Progress Bar */}
                {job.status === 'running' && (
                  <div className="mb-2">
                    <div className="flex justify-between text-sm mb-1">
                      <span>Progress</span>
                      <span>{job.progress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${job.progress}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Timing Info */}
                <div className="grid grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Created:</span>
                    <span className="ml-2 font-medium">{formatTime(job.created)}</span>
                  </div>
                  {job.started && (
                    <div>
                      <span className="text-gray-600">Started:</span>
                      <span className="ml-2 font-medium">{formatTime(job.started)}</span>
                    </div>
                  )}
                  {job.completed && (
                    <div>
                      <span className="text-gray-600">Completed:</span>
                      <span className="ml-2 font-medium">{formatTime(job.completed)}</span>
                    </div>
                  )}
                  {job.duration && (
                    <div>
                      <span className="text-gray-600">Duration:</span>
                      <span className="ml-2 font-medium">{formatDuration(job.duration)}</span>
                    </div>
                  )}
                </div>

                {/* Parameters */}
                <div className="mt-2 text-sm">
                  <span className="text-gray-600">Parameters:</span>
                  <span className="ml-2 font-mono text-xs">
                    {JSON.stringify(job.parameters)}
                  </span>
                </div>

                {/* Error Message */}
                {job.error && (
                  <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-800">
                    <strong>Error:</strong> {job.error}
                  </div>
                )}

                {/* Result */}
                {job.result && (
                  <div className="mt-2 p-2 bg-green-50 border border-green-200 rounded text-sm">
                    <strong>Result:</strong> {JSON.stringify(job.result)}
                  </div>
                )}
              </div>

              {/* Actions */}
              <div className="flex gap-2 ml-4">
                {job.status === 'failed' && (
                  <button
                    onClick={() => retryJob(job.id)}
                    className="p-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                    title="Retry"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>
                )}
                {(job.status === 'pending' || job.status === 'running') && (
                  <button
                    onClick={() => cancelJob(job.id)}
                    className="p-2 bg-red-600 text-white rounded hover:bg-red-700"
                    title="Cancel"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {filteredJobs.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          No jobs found with status: {filter}
        </div>
      )}
    </div>
  );
};

export default JobOrchestrationDashboard;
