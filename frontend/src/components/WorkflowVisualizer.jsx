/**
 * Workflow Visualizer
 * LangGraph workflow state and transitions visualization
 */

import React, { useState } from 'react';
import { GitBranch, Play, CheckCircle, Circle } from 'lucide-react';

const WorkflowVisualizer = () => {
  const [workflow] = useState({
    name: 'Aerodynamic Optimization Workflow',
    currentNode: 'quantum_agent',
    nodes: [
      { id: 'intent_router', name: 'Intent Router', status: 'completed', duration: 0.5 },
      { id: 'ml_agent', name: 'ML Surrogate Agent', status: 'completed', duration: 0.8 },
      { id: 'physics_agent', name: 'Physics Validator', status: 'completed', duration: 4.2 },
      { id: 'quantum_agent', name: 'Quantum Optimizer', status: 'running', duration: null },
      { id: 'analysis_agent', name: 'Analysis Agent', status: 'pending', duration: null },
      { id: 'report_agent', name: 'Report Generator', status: 'pending', duration: null }
    ],
    edges: [
      { from: 'intent_router', to: 'ml_agent' },
      { from: 'ml_agent', to: 'physics_agent' },
      { from: 'physics_agent', to: 'quantum_agent', condition: 'confidence < 0.9' },
      { from: 'quantum_agent', to: 'analysis_agent' },
      { from: 'analysis_agent', to: 'report_agent' }
    ],
    state: {
      user_query: 'Optimize wing for maximum downforce',
      mesh_id: 'wing_v3.2',
      ml_prediction: { Cl: 2.8, Cd: 0.42, confidence: 0.85 },
      physics_validation: { validated: true, error: 0.03 },
      quantum_optimization: { inProgress: true }
    }
  });

  const getNodeStatus = (status) => {
    switch (status) {
      case 'completed':
        return { icon: <CheckCircle className="w-5 h-5" />, color: 'bg-green-100 border-green-300 text-green-800' };
      case 'running':
        return { icon: <Play className="w-5 h-5 animate-pulse" />, color: 'bg-blue-100 border-blue-300 text-blue-800' };
      case 'pending':
        return { icon: <Circle className="w-5 h-5" />, color: 'bg-gray-100 border-gray-300 text-gray-600' };
      default:
        return { icon: <Circle className="w-5 h-5" />, color: 'bg-gray-100 border-gray-300 text-gray-600' };
    }
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
        <GitBranch className="w-6 h-6" />
        Workflow Visualizer
      </h2>

      <p className="text-gray-600 mb-6">
        LangGraph workflow execution with state transitions and agent handoffs.
      </p>

      {/* Workflow Info */}
      <div className="mb-6 p-4 bg-purple-50 border border-purple-200 rounded">
        <h3 className="font-semibold mb-2">{workflow.name}</h3>
        <div className="text-sm text-gray-700">
          <strong>Query:</strong> {workflow.state.user_query}
        </div>
        <div className="text-sm text-gray-700">
          <strong>Mesh:</strong> {workflow.state.mesh_id}
        </div>
      </div>

      {/* Workflow Graph */}
      <div className="space-y-4 mb-6">
        {workflow.nodes.map((node, idx) => {
          const status = getNodeStatus(node.status);
          const isActive = node.id === workflow.currentNode;
          
          return (
            <div key={node.id}>
              <div className={`p-4 rounded-lg border-2 ${status.color} ${isActive ? 'ring-2 ring-purple-500' : ''}`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {status.icon}
                    <div>
                      <div className="font-semibold">{node.name}</div>
                      <div className="text-xs opacity-75 capitalize">{node.status}</div>
                    </div>
                  </div>
                  {node.duration && (
                    <div className="text-sm">
                      <span className="text-gray-600">Duration:</span>
                      <span className="ml-2 font-mono font-bold">{node.duration.toFixed(1)}s</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Edge Arrow */}
              {idx < workflow.nodes.length - 1 && (
                <div className="flex justify-center my-2">
                  <div className="text-gray-400">↓</div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Current State */}
      <div className="p-4 bg-gray-50 rounded border border-gray-200">
        <h3 className="font-semibold mb-3">Current Workflow State</h3>
        <pre className="text-xs bg-white p-3 rounded border overflow-x-auto">
          {JSON.stringify(workflow.state, null, 2)}
        </pre>
      </div>
    </div>
  );
};

export default WorkflowVisualizer;
