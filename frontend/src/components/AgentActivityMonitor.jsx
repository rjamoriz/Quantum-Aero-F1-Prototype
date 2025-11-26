/**
 * Agent Activity Monitor
 * Visualize GenAI agent coordination and activity
 */

import React, { useState, useEffect } from 'react';
import { Bot, Activity, CheckCircle, Clock, AlertCircle } from 'lucide-react';

const AgentActivityMonitor = () => {
  const [agents, setAgents] = useState([]);

  useEffect(() => {
    loadAgentStatus();
    const interval = setInterval(loadAgentStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const loadAgentStatus = () => {
    // Mock agent status
    const mockAgents = [
      {
        name: 'Master Orchestrator',
        model: 'Claude Sonnet 4.5',
        status: 'active',
        tasksCompleted: 45,
        avgResponseTime: 2.3,
        currentTask: 'Coordinating optimization workflow',
        lastActive: new Date()
      },
      {
        name: 'Aerodynamics Agent',
        model: 'Claude Sonnet 4.5',
        status: 'idle',
        tasksCompleted: 23,
        avgResponseTime: 3.1,
        currentTask: null,
        lastActive: new Date(Date.now() - 120000)
      },
      {
        name: 'ML Surrogate Agent',
        model: 'Claude Haiku',
        status: 'active',
        tasksCompleted: 156,
        avgResponseTime: 0.8,
        currentTask: 'Running inference predictions',
        lastActive: new Date()
      },
      {
        name: 'Quantum Optimizer Agent',
        model: 'Claude Sonnet 4.5',
        status: 'idle',
        tasksCompleted: 12,
        avgResponseTime: 8.5,
        currentTask: null,
        lastActive: new Date(Date.now() - 300000)
      },
      {
        name: 'Physics Validator Agent',
        model: 'Claude Haiku',
        status: 'idle',
        tasksCompleted: 34,
        avgResponseTime: 4.2,
        currentTask: null,
        lastActive: new Date(Date.now() - 60000)
      },
      {
        name: 'Analysis Agent',
        model: 'Claude Sonnet 4.5',
        status: 'active',
        tasksCompleted: 28,
        avgResponseTime: 5.1,
        currentTask: 'Performing trade-off analysis',
        lastActive: new Date()
      }
    ];
    setAgents(mockAgents);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 border-green-300 text-green-800';
      case 'idle':
        return 'bg-gray-100 border-gray-300 text-gray-600';
      case 'error':
        return 'bg-red-100 border-red-300 text-red-800';
      default:
        return 'bg-gray-100 border-gray-300 text-gray-600';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'active':
        return <Activity className="w-5 h-5 text-green-600 animate-pulse" />;
      case 'idle':
        return <Clock className="w-5 h-5 text-gray-400" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-600" />;
      default:
        return <Bot className="w-5 h-5 text-gray-400" />;
    }
  };

  const activeAgents = agents.filter(a => a.status === 'active').length;
  const totalTasks = agents.reduce((sum, a) => sum + a.tasksCompleted, 0);

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
        <Bot className="w-6 h-6" />
        Agent Activity Monitor
      </h2>

      {/* Summary */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="p-4 bg-green-50 border border-green-200 rounded">
          <div className="text-sm text-green-700">Active Agents</div>
          <div className="text-3xl font-bold text-green-900">{activeAgents}/{agents.length}</div>
        </div>
        <div className="p-4 bg-blue-50 border border-blue-200 rounded">
          <div className="text-sm text-blue-700">Total Tasks</div>
          <div className="text-3xl font-bold text-blue-900">{totalTasks}</div>
        </div>
        <div className="p-4 bg-purple-50 border border-purple-200 rounded">
          <div className="text-sm text-purple-700">Avg Response</div>
          <div className="text-3xl font-bold text-purple-900">
            {(agents.reduce((sum, a) => sum + a.avgResponseTime, 0) / agents.length).toFixed(1)}s
          </div>
        </div>
      </div>

      {/* Agent Cards */}
      <div className="space-y-3">
        {agents.map((agent, idx) => (
          <div
            key={idx}
            className={`p-4 rounded-lg border-2 ${getStatusColor(agent.status)}`}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                {getStatusIcon(agent.status)}
                <div>
                  <div className="font-semibold text-lg">{agent.name}</div>
                  <div className="text-sm opacity-75">{agent.model}</div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs opacity-75">Status</div>
                <div className="font-semibold capitalize">{agent.status}</div>
              </div>
            </div>

            {agent.currentTask && (
              <div className="mt-3 p-2 bg-white rounded text-sm">
                <strong>Current Task:</strong> {agent.currentTask}
              </div>
            )}

            <div className="grid grid-cols-3 gap-4 mt-3 text-sm">
              <div>
                <div className="text-xs opacity-75">Tasks Completed</div>
                <div className="font-semibold">{agent.tasksCompleted}</div>
              </div>
              <div>
                <div className="text-xs opacity-75">Avg Response</div>
                <div className="font-semibold">{agent.avgResponseTime.toFixed(1)}s</div>
              </div>
              <div>
                <div className="text-xs opacity-75">Last Active</div>
                <div className="font-semibold">
                  {Math.floor((Date.now() - agent.lastActive) / 1000)}s ago
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AgentActivityMonitor;
