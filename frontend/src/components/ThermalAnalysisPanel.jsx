import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, AreaChart, Area, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine, Cell, BarChart, Bar
} from 'recharts';
import './ThermalAnalysisPanel.css';

/**
 * Thermal Analysis Panel
 * Conjugate Heat Transfer (CHT) analysis with thermal-structural coupling
 * - Temperature distribution visualization
 * - Heat flux analysis
 * - Thermal stress calculation
 * - Transient thermal response
 */
const ThermalAnalysisPanel = ({ modalData }) => {
  const [config, setConfig] = useState({
    component: 'front_wing',
    velocity: 300, // km/h
    ambientTemp: 25, // °C
    trackTemp: 45, // °C
    brakeTemp: 800, // °C (for heat sources)
    analysisType: 'steady', // steady or transient
    duration: 60, // seconds for transient
    material: 'carbon_fiber',
  });

  const [thermalResults, setThermalResults] = useState({
    temperatureDistribution: [],
    heatFluxData: [],
    thermalStress: [],
    transientResponse: [],
    statistics: null,
  });

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [displayOptions, setDisplayOptions] = useState({
    showTemperature: true,
    showHeatFlux: true,
    showThermalStress: true,
    showTransient: false,
  });
  const [logs, setLogs] = useState([]);

  // Component thermal properties
  const componentProperties = {
    front_wing: {
      length: 1.8,
      chord: 0.5,
      thickness: 0.002, // m
      surfaceArea: 0.9, // m²
      exposedArea: 1.8, // m² (both sides)
    },
    rear_wing: {
      length: 1.5,
      chord: 0.6,
      thickness: 0.0025,
      surfaceArea: 0.9,
      exposedArea: 1.8,
    },
    floor: {
      length: 3.5,
      chord: 1.2,
      thickness: 0.003,
      surfaceArea: 4.2,
      exposedArea: 4.2,
    },
    diffuser: {
      length: 1.0,
      chord: 1.5,
      thickness: 0.002,
      surfaceArea: 1.5,
      exposedArea: 3.0,
    },
  };

  // Material thermal properties
  const materialProperties = {
    carbon_fiber: {
      k: 150, // W/(m·K) - High conductivity carbon
      rho: 1600, // kg/m³
      cp: 710, // J/(kg·K)
      alpha: 1.0e-6, // Thermal expansion coefficient (1/K)
      emissivity: 0.85,
      E: 150e9, // Young's modulus (Pa)
      nu: 0.3, // Poisson's ratio
    },
    aluminum: {
      k: 237,
      rho: 2700,
      cp: 900,
      alpha: 23e-6,
      emissivity: 0.9,
      E: 70e9,
      nu: 0.33,
    },
    titanium: {
      k: 21.9,
      rho: 4500,
      cp: 520,
      alpha: 8.6e-6,
      emissivity: 0.7,
      E: 110e9,
      nu: 0.34,
    },
  };

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, { timestamp, message, type }]);
  };

  // Calculate convective heat transfer coefficient
  const calculateConvectionCoeff = (velocity, length) => {
    const V = velocity / 3.6; // Convert to m/s
    const rho = 1.225; // kg/m³
    const mu = 1.81e-5; // Pa·s
    const k_air = 0.0257; // W/(m·K)
    const Pr = 0.71; // Prandtl number for air
    
    // Reynolds number
    const Re = (rho * V * length) / mu;
    
    // Nusselt number (turbulent flow approximation)
    const Nu = 0.037 * Math.pow(Re, 0.8) * Math.pow(Pr, 1/3);
    
    // Convection coefficient
    const h = (Nu * k_air) / length;
    
    return h;
  };

  // Steady-state thermal analysis
  const performSteadyStateThermal = () => {
    addLog('🌡️ Performing steady-state thermal analysis...', 'info');
    
    const geom = componentProperties[config.component];
    const mat = materialProperties[config.material];
    
    // Convection coefficient
    const h = calculateConvectionCoeff(config.velocity, geom.length);
    addLog(`Convection coefficient: ${h.toFixed(1)} W/(m²·K)`, 'info');
    
    // Heat generation from aerodynamic heating (skin friction)
    const V = config.velocity / 3.6;
    const q_aero = 0.5 * 1.225 * V * V * V * 0.002; // W/m² (simplified)
    
    // Solar radiation (if applicable)
    const q_solar = 800; // W/m² peak solar
    const absorptivity = 0.7;
    const q_rad_in = q_solar * absorptivity;
    
    // Radiation heat loss (Stefan-Boltzmann)
    const sigma = 5.67e-8; // W/(m²·K⁴)
    
    // Temperature distribution along length (1D approximation)
    const numPoints = 50;
    const temperatureDistribution = [];
    
    for (let i = 0; i < numPoints; i++) {
      const x = (i / (numPoints - 1)) * geom.length;
      const x_norm = x / geom.length;
      
      // Base temperature from convection balance
      const T_conv = config.ambientTemp + (q_aero / h);
      
      // Temperature variation along chord (higher at leading edge)
      const T_local = T_conv + 15 * Math.exp(-5 * x_norm);
      
      // Heat flux
      const q_conv = h * (T_local - config.ambientTemp);
      const T_kelvin = T_local + 273.15;
      const q_rad_out = mat.emissivity * sigma * (Math.pow(T_kelvin, 4) - Math.pow(config.ambientTemp + 273.15, 4));
      const q_net = q_aero + q_rad_in - q_conv - q_rad_out;
      
      // Thermal stress (σ = E·α·ΔT / (1-ν))
      const deltaT = T_local - config.ambientTemp;
      const thermalStress = (mat.E * mat.alpha * deltaT) / (1 - mat.nu);
      
      temperatureDistribution.push({
        x: x * 1000, // Convert to mm
        x_norm,
        temperature: T_local,
        q_conv,
        q_rad_out,
        q_net,
        thermalStress: thermalStress / 1e6, // Convert to MPa
      });
    }
    
    // Calculate statistics
    const temps = temperatureDistribution.map(d => d.temperature);
    const stresses = temperatureDistribution.map(d => d.thermalStress);
    
    const statistics = {
      T_max: Math.max(...temps),
      T_min: Math.min(...temps),
      T_avg: temps.reduce((a, b) => a + b, 0) / temps.length,
      deltaT_max: Math.max(...temps) - config.ambientTemp,
      stress_max: Math.max(...stresses),
      q_total: q_aero * geom.exposedArea,
      h_conv: h,
    };
    
    addLog(`Max temperature: ${statistics.T_max.toFixed(1)} °C`, 'success');
    addLog(`Max thermal stress: ${statistics.stress_max.toFixed(1)} MPa`, 'success');
    
    return { temperatureDistribution, statistics };
  };

  // Transient thermal analysis
  const performTransientThermal = () => {
    addLog('⏱️ Performing transient thermal analysis...', 'info');
    
    const geom = componentProperties[config.component];
    const mat = materialProperties[config.material];
    
    // Thermal diffusivity
    const alpha_thermal = mat.k / (mat.rho * mat.cp);
    addLog(`Thermal diffusivity: ${(alpha_thermal * 1e6).toFixed(2)} mm²/s`, 'info');
    
    // Time step
    const dt = 1.0; // seconds
    const numSteps = Math.floor(config.duration / dt);
    
    // Initial condition
    let T = config.ambientTemp;
    const transientResponse = [];
    
    // Heat input variation (e.g., cornering, braking)
    for (let i = 0; i < numSteps; i++) {
      const t = i * dt;
      
      // Variable velocity (cornering profile)
      const V_kmh = config.velocity * (1 + 0.2 * Math.sin(2 * Math.PI * t / 30));
      const V = V_kmh / 3.6;
      const h = calculateConvectionCoeff(V_kmh, geom.length);
      
      // Heat input
      const q_in = 0.5 * 1.225 * V * V * V * 0.002;
      
      // Heat loss
      const q_out = h * (T - config.ambientTemp);
      
      // Thermal mass
      const mass = mat.rho * geom.surfaceArea * geom.thickness;
      const C_thermal = mass * mat.cp;
      
      // Temperature change
      const dT = ((q_in * geom.exposedArea - q_out * geom.exposedArea) * dt) / C_thermal;
      T = T + dT;
      
      // Thermal stress
      const deltaT = T - config.ambientTemp;
      const stress = (mat.E * mat.alpha * deltaT) / (1 - mat.nu) / 1e6;
      
      if (i % 5 === 0) { // Sample every 5 seconds
        transientResponse.push({
          time: t,
          temperature: T,
          velocity: V_kmh,
          heatInput: q_in,
          heatOutput: q_out,
          thermalStress: stress,
        });
      }
    }
    
    addLog(`Final temperature: ${T.toFixed(1)} °C after ${config.duration}s`, 'success');
    
    return transientResponse;
  };

  // Run complete thermal analysis
  const runThermalAnalysis = () => {
    setIsAnalyzing(true);
    setLogs([]);
    addLog('🚀 Starting thermal analysis...', 'info');
    
    try {
      // Steady-state analysis
      const { temperatureDistribution, statistics } = performSteadyStateThermal();
      
      // Heat flux data (extract from temperature distribution)
      const heatFluxData = temperatureDistribution.map(d => ({
        x: d.x,
        convection: d.q_conv,
        radiation: d.q_rad_out,
        net: d.q_net,
      }));
      
      // Thermal stress data
      const thermalStress = temperatureDistribution.map(d => ({
        x: d.x,
        stress: d.thermalStress,
        yieldStress: config.material === 'carbon_fiber' ? 600 : 
                     config.material === 'aluminum' ? 400 : 800,
        safetyFactor: (config.material === 'carbon_fiber' ? 600 : 
                       config.material === 'aluminum' ? 400 : 800) / Math.abs(d.thermalStress),
      }));
      
      let transientResponse = [];
      if (config.analysisType === 'transient') {
        transientResponse = performTransientThermal();
      }
      
      setThermalResults({
        temperatureDistribution,
        heatFluxData,
        thermalStress,
        transientResponse,
        statistics,
      });
      
      addLog('✅ Thermal analysis completed successfully!', 'success');
      
    } catch (error) {
      addLog(`❌ Analysis failed: ${error.message}`, 'error');
      console.error(error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Export results
  const exportResults = () => {
    const data = {
      config,
      results: thermalResults,
      timestamp: new Date().toISOString(),
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `thermal_analysis_${config.component}_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
    
    addLog('📥 Results exported successfully', 'success');
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="thermal-tooltip">
          <p className="tooltip-label">Position: {label} mm</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: {entry.value.toFixed(2)} {entry.unit || ''}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="thermal-analysis-panel">
      <div className="panel-header">
        <h2>🌡️ Thermal Analysis Panel</h2>
        <p>Conjugate Heat Transfer & Thermal-Structural Coupling</p>
      </div>

      {/* Configuration Section */}
      <div className="config-section">
        <h3>Configuration</h3>
        
        <div className="config-grid">
          <div className="config-item">
            <label>Component</label>
            <select
              value={config.component}
              onChange={(e) => setConfig({ ...config, component: e.target.value })}
            >
              <option value="front_wing">Front Wing</option>
              <option value="rear_wing">Rear Wing</option>
              <option value="floor">Floor</option>
              <option value="diffuser">Diffuser</option>
            </select>
          </div>

          <div className="config-item">
            <label>Material</label>
            <select
              value={config.material}
              onChange={(e) => setConfig({ ...config, material: e.target.value })}
            >
              <option value="carbon_fiber">Carbon Fiber</option>
              <option value="aluminum">Aluminum 7075</option>
              <option value="titanium">Titanium Ti-6Al-4V</option>
            </select>
          </div>

          <div className="config-item">
            <label>Velocity: {config.velocity} km/h</label>
            <input
              type="range"
              min="50"
              max="400"
              value={config.velocity}
              onChange={(e) => setConfig({ ...config, velocity: parseInt(e.target.value) })}
            />
          </div>

          <div className="config-item">
            <label>Ambient Temp: {config.ambientTemp} °C</label>
            <input
              type="range"
              min="0"
              max="50"
              value={config.ambientTemp}
              onChange={(e) => setConfig({ ...config, ambientTemp: parseInt(e.target.value) })}
            />
          </div>

          <div className="config-item">
            <label>Track Temp: {config.trackTemp} °C</label>
            <input
              type="range"
              min="20"
              max="70"
              value={config.trackTemp}
              onChange={(e) => setConfig({ ...config, trackTemp: parseInt(e.target.value) })}
            />
          </div>

          <div className="config-item">
            <label>Analysis Type</label>
            <select
              value={config.analysisType}
              onChange={(e) => setConfig({ ...config, analysisType: e.target.value })}
            >
              <option value="steady">Steady-State</option>
              <option value="transient">Transient</option>
            </select>
          </div>

          {config.analysisType === 'transient' && (
            <div className="config-item">
              <label>Duration: {config.duration} s</label>
              <input
                type="range"
                min="10"
                max="300"
                step="10"
                value={config.duration}
                onChange={(e) => setConfig({ ...config, duration: parseInt(e.target.value) })}
              />
            </div>
          )}
        </div>

        <div className="action-buttons">
          <button
            className="analyze-btn"
            onClick={runThermalAnalysis}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? '🔄 Analyzing...' : '▶️ Run Thermal Analysis'}
          </button>
          
          <button
            className="export-btn"
            onClick={exportResults}
            disabled={!thermalResults.statistics}
          >
            📥 Export Results
          </button>
        </div>
      </div>

      {/* Statistics Summary */}
      {thermalResults.statistics && (
        <div className="statistics-section">
          <h3>Thermal Statistics</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <span className="stat-icon">🌡️</span>
              <div className="stat-content">
                <span className="stat-label">Max Temperature</span>
                <span className="stat-value">{thermalResults.statistics.T_max.toFixed(1)} °C</span>
              </div>
            </div>

            <div className="stat-card">
              <span className="stat-icon">❄️</span>
              <div className="stat-content">
                <span className="stat-label">Min Temperature</span>
                <span className="stat-value">{thermalResults.statistics.T_min.toFixed(1)} °C</span>
              </div>
            </div>

            <div className="stat-card">
              <span className="stat-icon">📊</span>
              <div className="stat-content">
                <span className="stat-label">Avg Temperature</span>
                <span className="stat-value">{thermalResults.statistics.T_avg.toFixed(1)} °C</span>
              </div>
            </div>

            <div className="stat-card">
              <span className="stat-icon">🔥</span>
              <div className="stat-content">
                <span className="stat-label">ΔT Max</span>
                <span className="stat-value">{thermalResults.statistics.deltaT_max.toFixed(1)} K</span>
              </div>
            </div>

            <div className="stat-card">
              <span className="stat-icon">💪</span>
              <div className="stat-content">
                <span className="stat-label">Max Thermal Stress</span>
                <span className="stat-value">{thermalResults.statistics.stress_max.toFixed(1)} MPa</span>
              </div>
            </div>

            <div className="stat-card">
              <span className="stat-icon">⚡</span>
              <div className="stat-content">
                <span className="stat-label">Heat Transfer Coeff</span>
                <span className="stat-value">{thermalResults.statistics.h_conv.toFixed(1)} W/(m²·K)</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Results Visualization */}
      {thermalResults.temperatureDistribution.length > 0 && (
        <div className="results-section">
          <div className="display-options">
            <h3>Display Options</h3>
            <div className="options-grid">
              <label>
                <input
                  type="checkbox"
                  checked={displayOptions.showTemperature}
                  onChange={(e) => setDisplayOptions({ ...displayOptions, showTemperature: e.target.checked })}
                />
                Temperature Distribution
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={displayOptions.showHeatFlux}
                  onChange={(e) => setDisplayOptions({ ...displayOptions, showHeatFlux: e.target.checked })}
                />
                Heat Flux
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={displayOptions.showThermalStress}
                  onChange={(e) => setDisplayOptions({ ...displayOptions, showThermalStress: e.target.checked })}
                />
                Thermal Stress
              </label>
              {config.analysisType === 'transient' && (
                <label>
                  <input
                    type="checkbox"
                    checked={displayOptions.showTransient}
                    onChange={(e) => setDisplayOptions({ ...displayOptions, showTransient: e.target.checked })}
                  />
                  Transient Response
                </label>
              )}
            </div>
          </div>

          {/* Temperature Distribution */}
          {displayOptions.showTemperature && (
            <div className="chart-container">
              <h4>Temperature Distribution</h4>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={thermalResults.temperatureDistribution}>
                  <defs>
                    <linearGradient id="tempGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ff6b6b" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#ff6b6b" stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="x" stroke="#fff" label={{ value: 'Position (mm)', position: 'insideBottom', offset: -5 }} />
                  <YAxis stroke="#fff" label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <ReferenceLine y={config.ambientTemp} stroke="#00c8ff" strokeDasharray="3 3" label="Ambient" />
                  <Area type="monotone" dataKey="temperature" stroke="#ff6b6b" fill="url(#tempGradient)" name="Temperature" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Heat Flux */}
          {displayOptions.showHeatFlux && (
            <div className="chart-container">
              <h4>Heat Flux Distribution</h4>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={thermalResults.heatFluxData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="x" stroke="#fff" label={{ value: 'Position (mm)', position: 'insideBottom', offset: -5 }} />
                  <YAxis stroke="#fff" label={{ value: 'Heat Flux (W/m²)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line type="monotone" dataKey="convection" stroke="#00c8ff" name="Convection" strokeWidth={2} />
                  <Line type="monotone" dataKey="radiation" stroke="#ff00ff" name="Radiation" strokeWidth={2} />
                  <Line type="monotone" dataKey="net" stroke="#00ff88" name="Net Flux" strokeWidth={2} />
                  <ReferenceLine y={0} stroke="#fff" strokeDasharray="3 3" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Thermal Stress */}
          {displayOptions.showThermalStress && (
            <div className="chart-container">
              <h4>Thermal Stress Distribution</h4>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={thermalResults.thermalStress}>
                  <defs>
                    <linearGradient id="stressGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ff8800" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#ff8800" stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="x" stroke="#fff" label={{ value: 'Position (mm)', position: 'insideBottom', offset: -5 }} />
                  <YAxis stroke="#fff" label={{ value: 'Stress (MPa)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <ReferenceLine y={thermalResults.thermalStress[0]?.yieldStress} stroke="#ff3366" strokeDasharray="3 3" label="Yield Limit" />
                  <Area type="monotone" dataKey="stress" stroke="#ff8800" fill="url(#stressGradient)" name="Thermal Stress" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Transient Response */}
          {displayOptions.showTransient && thermalResults.transientResponse.length > 0 && (
            <div className="chart-container">
              <h4>Transient Thermal Response</h4>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={thermalResults.transientResponse}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="time" stroke="#fff" label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }} />
                  <YAxis yAxisId="temp" stroke="#fff" label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft' }} />
                  <YAxis yAxisId="stress" orientation="right" stroke="#ff8800" label={{ value: 'Stress (MPa)', angle: 90, position: 'insideRight' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line yAxisId="temp" type="monotone" dataKey="temperature" stroke="#ff6b6b" name="Temperature" strokeWidth={2} />
                  <Line yAxisId="stress" type="monotone" dataKey="thermalStress" stroke="#ff8800" name="Thermal Stress" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Logs Section */}
      <div className="logs-section">
        <h3>Analysis Logs</h3>
        <div className="logs-container">
          {logs.map((log, index) => (
            <div key={index} className={`log-entry log-${log.type}`}>
              <span className="log-timestamp">[{log.timestamp}]</span>
              <span className="log-message">{log.message}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ThermalAnalysisPanel;
