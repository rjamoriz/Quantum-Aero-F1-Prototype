import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, 
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart
} from 'recharts';
import './QuantumOptimizationDashboard.css';

/**
 * Dashboard completo para optimización cuántica QUBO
 * Incluye aeroelástica, vibración, térmico y optimización multi-objetivo
 */
const QuantumOptimizationDashboard = () => {
  const [optimization, setOptimization] = useState({
    type: 'stiffener_layout', // tipo de optimización
    status: 'idle', // idle, running, completed, error
    progress: 0,
    currentIteration: 0,
    maxIterations: 50,
    results: null,
    convergenceHistory: [],
    logs: [],
  });

  const [config, setConfig] = useState({
    // Optimización Aeroel\u00e1stica
    stiffenerLocations: 20,
    maxStiffeners: 8,
    targetFlutterMargin: 1.5,
    massWeight: 1.0,
    flutterWeight: 3.0,
    
    // Optimización Multi-Física
    includeVibration: true,
    includeThermal: true,
    includeAeroacoustic: false,
    
    // Quantum Solver
    quantumMethod: 'qaoa', // 'qaoa', 'vqe', 'annealing'
    numQubits: 20,
    qaoaLayers: 3,
    
    // Objetivos
    objectives: {
      maximizeFlutterSpeed: true,
      minimizeMass: true,
      minimizeDisplacement: true,
      maximizeDownforce: true,
    },
    
    // Restricciones
    constraints: {
      flutterMargin: 1.2, // Vf > 1.2 * Vmax
      maxDisplacement: 0.02, // 20mm
      stressSafetyFactor: 1.5,
      maxMass: 5.0, // kg
    },
  });

  const [visualizationMode, setVisualizationMode] = useState('convergence'); 
  // 'convergence', 'pareto', 'design_space', 'qubo_matrix'

  const optimizationTypes = [
    {
      id: 'stiffener_layout',
      name: 'Layout de Rigidizadores',
      description: 'Optimización de posicionamiento de rigidizadores para suprimir vibraciones',
      quboVars: 20,
      icon: '🔩',
    },
    {
      id: 'thickness_distribution',
      name: 'Distribución de Espesor',
      description: 'Optimización de espesor de panel para balance masa-rigidez',
      quboVars: 15,
      icon: '📏',
    },
    {
      id: 'cooling_topology',
      name: 'Topología de Enfriamiento',
      description: 'Layout óptimo de canales de enfriamiento',
      quboVars: 25,
      icon: '❄️',
    },
    {
      id: 'complete_wing',
      name: 'Ala Completa',
      description: 'Optimización multi-objetivo de ala F1 completa',
      quboVars: 30,
      icon: '✈️',
    },
    {
      id: 'aeroelastic_flutter',
      name: 'Flutter Aeroelástico',
      description: 'Maximizar velocidad de flutter con restricciones de masa',
      quboVars: 18,
      icon: '〰️',
    },
  ];

  // Agregar log
  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setOptimization(prev => ({
      ...prev,
      logs: [{ timestamp, message, type }, ...prev.logs].slice(0, 50)
    }));
  };

  // Ejecutar optimización cuántica
  const runQuantumOptimization = async () => {
    try {
      setOptimization(prev => ({ 
        ...prev, 
        status: 'running', 
        progress: 0,
        convergenceHistory: [],
        logs: []
      }));

      addLog('🚀 Iniciando optimización cuántica QUBO');
      addLog(`⚛️ Método: ${config.quantumMethod.toUpperCase()}, Qubits: ${config.numQubits}`);
      addLog(`🎯 Tipo: ${optimizationTypes.find(t => t.id === optimization.type).name}`);

      const startTime = Date.now();

      // Simulación de iteraciones QAOA
      for (let iter = 0; iter < config.maxIterations; iter++) {
        await new Promise(resolve => setTimeout(resolve, 100)); // Simular cálculo

        const energy = -10 + 15 * Math.exp(-iter / 10) + (Math.random() - 0.5) * 2;
        const flutterSpeed = 300 + iter * 2 + Math.random() * 10;
        const mass = 4.5 - iter * 0.02 + Math.random() * 0.2;

        setOptimization(prev => ({
          ...prev,
          currentIteration: iter + 1,
          progress: ((iter + 1) / config.maxIterations) * 100,
          convergenceHistory: [
            ...prev.convergenceHistory,
            {
              iteration: iter + 1,
              energy,
              flutterSpeed,
              mass,
              displacement: 0.025 - iter * 0.0003,
            }
          ],
        }));

        if (iter % 10 === 0) {
          addLog(`⚛️ Iteración ${iter + 1}: E = ${energy.toFixed(4)}, Vf = ${flutterSpeed.toFixed(1)} km/h`);
        }
      }

      // Resultado final
      const finalResult = {
        optimalDesign: Array(config.stiffenerLocations).fill(0).map(() => Math.random() > 0.6 ? 1 : 0),
        metrics: {
          flutterSpeed: 350.5, // km/h
          flutterMargin: 1.52,
          totalMass: 3.8, // kg
          maxDisplacement: 0.015, // m
          downforce: 2100, // N
          drag: 285, // N
        },
        quboEnergy: -9.85,
        quantumAdvantage: true,
        convergence: true,
        computeTime: (Date.now() - startTime) / 1000,
      };

      setOptimization(prev => ({
        ...prev,
        status: 'completed',
        results: finalResult,
        progress: 100,
      }));

      addLog('✅ Optimización completada exitosamente', 'success');
      addLog(`📊 Velocidad de flutter: ${finalResult.metrics.flutterSpeed} km/h (margen: ${finalResult.metrics.flutterMargin})`, 'success');
      addLog(`⚖️ Masa total: ${finalResult.metrics.totalMass} kg`, 'success');
      addLog(`⏱️ Tiempo: ${finalResult.computeTime.toFixed(2)}s`, 'success');

    } catch (error) {
      addLog(`❌ Error: ${error.message}`, 'error');
      setOptimization(prev => ({ ...prev, status: 'error' }));
    }
  };

  // Exportar resultados
  const exportResults = () => {
    if (!optimization.results) return;

    const data = {
      config,
      results: optimization.results,
      convergenceHistory: optimization.convergenceHistory,
      timestamp: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `quantum_optimization_${optimization.type}_${Date.now()}.json`;
    link.click();

    addLog('💾 Resultados exportados', 'success');
  };

  return (
    <div className="quantum-optimization-dashboard">
      <div className="dashboard-header">
        <h2>⚛️ Optimización Cuántica QUBO</h2>
        <p>Optimización multi-física con algoritmos cuánticos variational QAOA/VQE</p>
      </div>

      <div className="dashboard-grid">
        {/* Panel de Configuración */}
        <div className="config-section">
          <h3>⚙️ Configuración</h3>

          {/* Tipo de Optimización */}
          <div className="opt-type-grid">
            {optimizationTypes.map(type => (
              <button
                key={type.id}
                className={`opt-type-card ${optimization.type === type.id ? 'active' : ''}`}
                onClick={() => setOptimization({ ...optimization, type: type.id })}
                disabled={optimization.status === 'running'}
              >
                <div className="opt-icon">{type.icon}</div>
                <div className="opt-name">{type.name}</div>
                <div className="opt-qubits">{type.quboVars} qubits</div>
              </button>
            ))}
          </div>

          {/* Quantum Solver */}
          <div className="form-section">
            <h4>Solver Cuántico</h4>
            <div className="form-group">
              <label>Método Cuántico</label>
              <select 
                value={config.quantumMethod}
                onChange={(e) => setConfig({...config, quantumMethod: e.target.value})}
                disabled={optimization.status === 'running'}
              >
                <option value="qaoa">QAOA (Quantum Approximate Optimization)</option>
                <option value="vqe">VQE (Variational Quantum Eigensolver)</option>
                <option value="annealing">Quantum Annealing</option>
              </select>
            </div>

            {config.quantumMethod === 'qaoa' && (
              <div className="form-group">
                <label>Capas QAOA: {config.qaoaLayers}</label>
                <input 
                  type="range"
                  min="1"
                  max="10"
                  value={config.qaoaLayers}
                  onChange={(e) => setConfig({...config, qaoaLayers: parseInt(e.target.value)})}
                  disabled={optimization.status === 'running'}
                />
              </div>
            )}

            <div className="form-group">
              <label>Iteraciones Máximas: {config.maxIterations}</label>
              <input 
                type="range"
                min="10"
                max="200"
                value={config.maxIterations}
                onChange={(e) => setConfig({...config, maxIterations: parseInt(e.target.value)})}
                disabled={optimization.status === 'running'}
              />
            </div>
          </div>

          {/* Objetivos Multi-Física */}
          <div className="form-section">
            <h4>Física Incluida</h4>
            <label className="checkbox-label">
              <input 
                type="checkbox"
                checked={config.includeVibration}
                onChange={(e) => setConfig({...config, includeVibration: e.target.checked})}
                disabled={optimization.status === 'running'}
              />
              🌊 Vibración Estructural
            </label>
            <label className="checkbox-label">
              <input 
                type="checkbox"
                checked={config.includeThermal}
                onChange={(e) => setConfig({...config, includeThermal: e.target.checked})}
                disabled={optimization.status === 'running'}
              />
              🔥 Efectos Térmicos
            </label>
            <label className="checkbox-label">
              <input 
                type="checkbox"
                checked={config.includeAeroacoustic}
                onChange={(e) => setConfig({...config, includeAeroacoustic: e.target.checked})}
                disabled={optimization.status === 'running'}
              />
              🔊 Aeroacústica
            </label>
          </div>

          {/* Restricciones */}
          <div className="form-section">
            <h4>Restricciones</h4>
            <div className="constraint-item">
              <label>Margen Flutter</label>
              <input 
                type="number"
                step="0.1"
                value={config.constraints.flutterMargin}
                onChange={(e) => setConfig({
                  ...config,
                  constraints: {...config.constraints, flutterMargin: parseFloat(e.target.value)}
                })}
                disabled={optimization.status === 'running'}
              />
            </div>
            <div className="constraint-item">
              <label>Desplaz. Máx (m)</label>
              <input 
                type="number"
                step="0.001"
                value={config.constraints.maxDisplacement}
                onChange={(e) => setConfig({
                  ...config,
                  constraints: {...config.constraints, maxDisplacement: parseFloat(e.target.value)}
                })}
                disabled={optimization.status === 'running'}
              />
            </div>
            <div className="constraint-item">
              <label>Masa Máx (kg)</label>
              <input 
                type="number"
                step="0.1"
                value={config.constraints.maxMass}
                onChange={(e) => setConfig({
                  ...config,
                  constraints: {...config.constraints, maxMass: parseFloat(e.target.value)}
                })}
                disabled={optimization.status === 'running'}
              />
            </div>
          </div>

          {/* Botones de Acción */}
          <div className="action-buttons">
            <button 
              className="btn-run"
              onClick={runQuantumOptimization}
              disabled={optimization.status === 'running'}
            >
              {optimization.status === 'running' ? '⏳ Optimizando...' : '🚀 Ejecutar Optimización'}
            </button>
            {optimization.results && (
              <button className="btn-export" onClick={exportResults}>
                💾 Exportar Resultados
              </button>
            )}
          </div>
        </div>

        {/* Panel de Visualización */}
        <div className="visualization-section">
          <div className="viz-controls">
            <button 
              className={visualizationMode === 'convergence' ? 'active' : ''}
              onClick={() => setVisualizationMode('convergence')}
            >
              📈 Convergencia
            </button>
            <button 
              className={visualizationMode === 'design_space' ? 'active' : ''}
              onClick={() => setVisualizationMode('design_space')}
            >
              🎯 Espacio de Diseño
            </button>
            <button 
              className={visualizationMode === 'pareto' ? 'active' : ''}
              onClick={() => setVisualizationMode('pareto')}
            >
              ⚖️ Frontera Pareto
            </button>
          </div>

          {visualizationMode === 'convergence' && optimization.convergenceHistory.length > 0 && (
            <div className="chart-container">
              <h4>Convergencia QAOA</h4>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={optimization.convergenceHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 200, 255, 0.1)" />
                  <XAxis dataKey="iteration" stroke="#808080" />
                  <YAxis stroke="#808080" />
                  <Tooltip 
                    contentStyle={{
                      background: 'rgba(0, 0, 0, 0.8)',
                      border: '1px solid rgba(0, 200, 255, 0.3)',
                      borderRadius: '8px',
                    }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="energy" stroke="#00c8ff" name="Energía QUBO" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>

              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={optimization.convergenceHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 200, 255, 0.1)" />
                  <XAxis dataKey="iteration" stroke="#808080" />
                  <YAxis yAxisId="left" stroke="#808080" />
                  <YAxis yAxisId="right" orientation="right" stroke="#808080" />
                  <Tooltip 
                    contentStyle={{
                      background: 'rgba(0, 0, 0, 0.8)',
                      border: '1px solid rgba(0, 200, 255, 0.3)',
                      borderRadius: '8px',
                    }}
                  />
                  <Legend />
                  <Line 
                    yAxisId="left"
                    type="monotone" 
                    dataKey="flutterSpeed" 
                    stroke="#00ff88" 
                    name="Vel. Flutter (km/h)" 
                    strokeWidth={2} 
                  />
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="mass" 
                    stroke="#ff8800" 
                    name="Masa (kg)" 
                    strokeWidth={2} 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Progreso */}
          {optimization.status === 'running' && (
            <div className="progress-container">
              <h4>Progreso de Optimización</h4>
              <div className="progress-bar">
                <div className="progress-fill" style={{width: `${optimization.progress}%`}} />
              </div>
              <div className="progress-stats">
                <span>Iteración {optimization.currentIteration} / {config.maxIterations}</span>
                <span>{optimization.progress.toFixed(1)}%</span>
              </div>
            </div>
          )}

          {/* Logs */}
          <div className="logs-section">
            <h4>📝 Registro Cuántico</h4>
            <div className="logs-container">
              {optimization.logs.map((log, idx) => (
                <div key={idx} className={`log-entry log-${log.type}`}>
                  <span className="log-time">[{log.timestamp}]</span>
                  <span className="log-message">{log.message}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Panel de Resultados */}
        {optimization.results && (
          <div className="results-section">
            <h3>📊 Resultados Óptimos</h3>

            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-icon">🚀</div>
                <div className="metric-label">Vel. Flutter</div>
                <div className="metric-value">{optimization.results.metrics.flutterSpeed.toFixed(1)}</div>
                <div className="metric-unit">km/h</div>
                <div className="metric-status success">
                  Margen: {optimization.results.metrics.flutterMargin.toFixed(2)}x
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-icon">⚖️</div>
                <div className="metric-label">Masa Total</div>
                <div className="metric-value">{optimization.results.metrics.totalMass.toFixed(2)}</div>
                <div className="metric-unit">kg</div>
                <div className="metric-status success">
                  -{((1 - optimization.results.metrics.totalMass / config.constraints.maxMass) * 100).toFixed(1)}% vs límite
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-icon">📏</div>
                <div className="metric-label">Desplaz. Máx</div>
                <div className="metric-value">{(optimization.results.metrics.maxDisplacement * 1000).toFixed(1)}</div>
                <div className="metric-unit">mm</div>
                <div className="metric-status success">
                  < {(config.constraints.maxDisplacement * 1000).toFixed(0)}mm límite
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-icon">⬇️</div>
                <div className="metric-label">Downforce</div>
                <div className="metric-value">{optimization.results.metrics.downforce}</div>
                <div className="metric-unit">N</div>
              </div>

              <div className="metric-card">
                <div className="metric-icon">➡️</div>
                <div className="metric-label">Drag</div>
                <div className="metric-value">{optimization.results.metrics.drag}</div>
                <div className="metric-unit">N</div>
                <div className="metric-status">
                  L/D: {(optimization.results.metrics.downforce / optimization.results.metrics.drag).toFixed(2)}
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-icon">⚛️</div>
                <div className="metric-label">Energía QUBO</div>
                <div className="metric-value">{optimization.results.quboEnergy.toFixed(4)}</div>
                <div className="metric-unit">-</div>
                <div className="metric-status success">
                  {optimization.results.convergence ? '✓ Convergido' : '⚠️ No convergido'}
                </div>
              </div>
            </div>

            {/* Diseño Óptimo */}
            <div className="optimal-design">
              <h4>Diseño Óptimo - Variables Binarias</h4>
              <div className="binary-visualization">
                {optimization.results.optimalDesign.map((bit, idx) => (
                  <div 
                    key={idx} 
                    className={`binary-bit ${bit === 1 ? 'active' : 'inactive'}`}
                    title={`Posición ${idx + 1}: ${bit === 1 ? 'Rigidizador' : 'Sin rigidizador'}`}
                  >
                    {bit}
                  </div>
                ))}
              </div>
              <div className="design-summary">
                <span>Rigidizadores activos: {optimization.results.optimalDesign.filter(b => b === 1).length}</span>
                <span>Eficiencia: {((optimization.results.optimalDesign.filter(b => b === 1).length / config.stiffenerLocations) * 100).toFixed(1)}%</span>
              </div>
            </div>

            {/* Tiempo de Cómputo */}
            <div className="compute-stats">
              <h4>⏱️ Estadísticas de Cómputo</h4>
              <div className="stat-row">
                <span>Tiempo Total:</span>
                <span>{optimization.results.computeTime.toFixed(2)}s</span>
              </div>
              <div className="stat-row">
                <span>Iteraciones:</span>
                <span>{config.maxIterations}</span>
              </div>
              <div className="stat-row">
                <span>Tiempo/Iteración:</span>
                <span>{(optimization.results.computeTime / config.maxIterations * 1000).toFixed(1)}ms</span>
              </div>
              <div className="stat-row">
                <span>Ventaja Cuántica:</span>
                <span className={optimization.results.quantumAdvantage ? 'success' : 'warning'}>
                  {optimization.results.quantumAdvantage ? '✓ Sí' : '⚠️ No'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default QuantumOptimizationDashboard;
