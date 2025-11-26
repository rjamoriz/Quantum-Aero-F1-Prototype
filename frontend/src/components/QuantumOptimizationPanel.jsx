/**
 * Quantum Optimization Panel
 * Controls quantum-integrated multi-physics optimization
 */

import React, { useState } from 'react';
import axios from 'axios';

const QuantumOptimizationPanel = () => {
  const [optimizationType, setOptimizationType] = useState('complete_car');
  const [config, setConfig] = useState({
    use_quantum: true,
    use_ml_surrogate: true,
    include_vibration: true,
    include_thermal: true,
    include_acoustic: true,
    include_transient: true,
    n_iterations: 10
  });
  
  const [status, setStatus] = useState('idle');
  const [results, setResults] = useState(null);
  const [currentIteration, setCurrentIteration] = useState(0);

  const optimizationTypes = [
    { value: 'front_wing', label: 'Front Wing Optimization' },
    { value: 'rear_wing', label: 'Rear Wing Optimization' },
    { value: 'complete_car', label: 'Complete Car Optimization' },
    { value: 'stiffener_layout', label: 'Stiffener Layout (Vibration)' },
    { value: 'cooling_topology', label: 'Cooling Topology (Thermal)' },
    { value: 'transient_design', label: 'Transient Performance' }
  ];

  const startOptimization = async () => {
    setStatus('running');
    setCurrentIteration(0);
    
    try {
      let endpoint = '';
      let payload = {};

      switch (optimizationType) {
        case 'front_wing':
          endpoint = '/api/quantum/optimize-wing';
          payload = {
            wing_type: 'front',
            objectives: ['maximize_downforce', 'minimize_drag'],
            use_quantum: config.use_quantum,
            n_iterations: config.n_iterations
          };
          break;

        case 'rear_wing':
          endpoint = '/api/quantum/optimize-wing';
          payload = {
            wing_type: 'rear',
            objectives: ['maximize_downforce', 'minimize_drag'],
            use_quantum: config.use_quantum,
            n_iterations: config.n_iterations
          };
          break;

        case 'complete_car':
          endpoint = '/api/quantum/optimize-complete-car';
          payload = {
            objectives: ['maximize_downforce', 'minimize_drag', 'optimize_balance'],
            include_aeroelastic: true,
            include_transient: config.include_transient,
            use_quantum: config.use_quantum,
            n_iterations: config.n_iterations
          };
          break;

        case 'stiffener_layout':
          endpoint = '/api/quantum/optimize-stiffener-layout';
          payload = {
            n_locations: 20,
            max_stiffeners: 8,
            target_frequency: 50.0,
            use_quantum: config.use_quantum
          };
          break;

        case 'cooling_topology':
          endpoint = '/api/quantum/optimize-cooling-topology';
          payload = {
            grid_size: [10, 10, 5],
            max_temperature: 1000.0,
            use_quantum: config.use_quantum
          };
          break;

        case 'transient_design':
          endpoint = '/api/quantum/optimize-transient';
          payload = {
            scenario: 'corner_exit',
            include_vibration: config.include_vibration,
            include_thermal: config.include_thermal,
            include_acoustic: config.include_acoustic,
            n_iterations: config.n_iterations
          };
          break;
      }

      // Start optimization
      const response = await axios.post(`http://localhost:3001${endpoint}`, payload);
      
      setResults(response.data);
      setStatus('completed');

    } catch (error) {
      setStatus('error');
      console.error('Optimization error:', error);
    }
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4">Quantum Optimization</h2>

      {/* Optimization Type Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium mb-2">Optimization Type</label>
        <select
          value={optimizationType}
          onChange={(e) => setOptimizationType(e.target.value)}
          className="w-full px-3 py-2 border rounded"
        >
          {optimizationTypes.map(type => (
            <option key={type.value} value={type.value}>
              {type.label}
            </option>
          ))}
        </select>
      </div>

      {/* Configuration */}
      <div className="mb-6 p-4 bg-gray-50 rounded">
        <h3 className="font-semibold mb-3">Configuration</h3>
        
        <div className="space-y-3">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={config.use_quantum}
              onChange={(e) => setConfig({...config, use_quantum: e.target.checked})}
              className="mr-2"
            />
            <span className="text-sm">Use Quantum Solver (QAOA)</span>
          </label>

          <label className="flex items-center">
            <input
              type="checkbox"
              checked={config.use_ml_surrogate}
              onChange={(e) => setConfig({...config, use_ml_surrogate: e.target.checked})}
              className="mr-2"
            />
            <span className="text-sm">Use ML Surrogate (Fast Evaluation)</span>
          </label>

          <label className="flex items-center">
            <input
              type="checkbox"
              checked={config.include_vibration}
              onChange={(e) => setConfig({...config, include_vibration: e.target.checked})}
              className="mr-2"
            />
            <span className="text-sm">Include Vibration Analysis</span>
          </label>

          <label className="flex items-center">
            <input
              type="checkbox"
              checked={config.include_thermal}
              onChange={(e) => setConfig({...config, include_thermal: e.target.checked})}
              className="mr-2"
            />
            <span className="text-sm">Include Thermal Analysis</span>
          </label>

          <label className="flex items-center">
            <input
              type="checkbox"
              checked={config.include_acoustic}
              onChange={(e) => setConfig({...config, include_acoustic: e.target.checked})}
              className="mr-2"
            />
            <span className="text-sm">Include Acoustic Analysis</span>
          </label>

          <label className="flex items-center">
            <input
              type="checkbox"
              checked={config.include_transient}
              onChange={(e) => setConfig({...config, include_transient: e.target.checked})}
              className="mr-2"
            />
            <span className="text-sm">Include Transient Effects</span>
          </label>

          <div>
            <label className="block text-sm font-medium mb-1">Number of Iterations</label>
            <input
              type="number"
              value={config.n_iterations}
              onChange={(e) => setConfig({...config, n_iterations: parseInt(e.target.value)})}
              className="w-full px-3 py-2 border rounded"
              min="1"
              max="100"
            />
          </div>
        </div>
      </div>

      {/* Start Button */}
      <button
        onClick={startOptimization}
        disabled={status === 'running'}
        className={`w-full px-6 py-3 rounded font-semibold mb-6 ${
          status === 'running'
            ? 'bg-gray-400 cursor-not-allowed'
            : 'bg-purple-600 hover:bg-purple-700 text-white'
        }`}
      >
        {status === 'running' ? `Optimizing... (${currentIteration}/${config.n_iterations})` : 'Start Optimization'}
      </button>

      {/* Results Display */}
      {results && (
        <div className="space-y-4">
          <div className="p-4 bg-purple-50 border border-purple-200 rounded">
            <h3 className="font-semibold mb-3 text-purple-800">Optimization Results</h3>
            
            {/* Best Solution */}
            <div className="mb-4">
              <h4 className="font-medium mb-2">Best Solution</h4>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="p-2 bg-white rounded">
                  <span className="font-medium">Fitness:</span>
                  <span className="ml-2">{results.best_fitness?.toFixed(4)}</span>
                </div>
                <div className="p-2 bg-white rounded">
                  <span className="font-medium">Iterations:</span>
                  <span className="ml-2">{results.n_iterations}</span>
                </div>
              </div>
            </div>

            {/* Performance Metrics */}
            {results.best_solution?.performance && (
              <div className="mb-4">
                <h4 className="font-medium mb-2">Performance</h4>
                <div className="grid grid-cols-3 gap-3 text-sm">
                  {Object.entries(results.best_solution.performance).map(([key, value]) => (
                    <div key={key} className="p-2 bg-white rounded">
                      <div className="font-medium text-xs text-gray-600 mb-1">
                        {key.replace(/_/g, ' ').toUpperCase()}
                      </div>
                      <div className="text-lg font-bold">
                        {typeof value === 'number' ? value.toFixed(2) : value}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Multi-Physics Results */}
            {results.best_solution?.multiphysics && (
              <div>
                <h4 className="font-medium mb-2">Multi-Physics Analysis</h4>
                
                {results.best_solution.multiphysics.vibration && (
                  <div className="mb-2 p-2 bg-white rounded text-sm">
                    <span className="font-medium">Vibration:</span>
                    <span className={`ml-2 ${results.best_solution.multiphysics.vibration.safe ? 'text-green-600' : 'text-red-600'}`}>
                      Flutter Margin: {results.best_solution.multiphysics.vibration.flutter_margin?.toFixed(2)}
                      {results.best_solution.multiphysics.vibration.safe ? ' ✓' : ' ✗'}
                    </span>
                  </div>
                )}

                {results.best_solution.multiphysics.thermal && (
                  <div className="mb-2 p-2 bg-white rounded text-sm">
                    <span className="font-medium">Thermal:</span>
                    <span className={`ml-2 ${results.best_solution.multiphysics.thermal.safe ? 'text-green-600' : 'text-red-600'}`}>
                      Brake Temp: {results.best_solution.multiphysics.thermal.brake_temperature?.toFixed(0)}K
                      {results.best_solution.multiphysics.thermal.safe ? ' ✓' : ' ✗'}
                    </span>
                  </div>
                )}

                {results.best_solution.multiphysics.acoustic && (
                  <div className="p-2 bg-white rounded text-sm">
                    <span className="font-medium">Acoustic:</span>
                    <span className={`ml-2 ${results.best_solution.multiphysics.acoustic.compliant ? 'text-green-600' : 'text-red-600'}`}>
                      SPL: {results.best_solution.multiphysics.acoustic.spl?.toFixed(1)} dB
                      {results.best_solution.multiphysics.acoustic.compliant ? ' ✓' : ' ✗'}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Convergence History */}
          {results.history && (
            <div className="p-4 bg-gray-50 rounded">
              <h4 className="font-medium mb-2">Convergence History</h4>
              <div className="h-48 flex items-end space-x-1">
                {results.history.map((point, idx) => (
                  <div
                    key={idx}
                    className="flex-1 bg-purple-600 rounded-t"
                    style={{
                      height: `${(1 - point.fitness / results.history[0].fitness) * 100}%`,
                      minHeight: '2px'
                    }}
                    title={`Iteration ${point.iteration}: ${point.fitness.toFixed(4)}`}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default QuantumOptimizationPanel;
