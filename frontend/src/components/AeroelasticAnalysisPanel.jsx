import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  LineChart, Line, ScatterChart, Scatter, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine, ReferenceArea
} from 'recharts';
import './AeroelasticAnalysisPanel.css';

/**
 * Aeroelastic Analysis Panel
 * Comprehensive aeroelastic analysis with flutter prediction, modal analysis,
 * and high-speed aerodynamic load generation for F1 airfoils
 */
const AeroelasticAnalysisPanel = ({ onModalDataGenerated }) => {
  const [config, setConfig] = useState({
    component: 'front_wing',
    nacaProfile: 'NACA6412',
    velocity: 300, // km/h
    yawAngle: 0, // degrees
    structuralConfig: {
      thickness: 2.0, // mm
      stiffenerCount: 4,
      material: 'carbon_fiber',
    },
  });

  const [analysisResults, setAnalysisResults] = useState({
    modalData: {
      frequencies: [],
      dampingRatios: [],
      modeShapes: [],
    },
    flutterData: {
      flutterSpeed: 0,
      flutterMargin: 0,
      criticalMode: null,
      vgData: [], // Velocity-damping (V-g) diagram
    },
    aeroLoads: {
      staticLoads: [],
      dynamicLoads: [],
      pressureDistribution: [],
    },
    deformation: {
      maxDisplacement: 0,
      tipDeflection: 0,
      twistAngle: 0,
    },
  });

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [logs, setLogs] = useState([]);
  const [selectedMode, setSelectedMode] = useState(0);

  // Add log entry
  const addLog = (message, type = 'info', data = null) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [
      { timestamp, message, type, data },
      ...prev
    ].slice(0, 100));
  };

  // Generate high-speed aerodynamic loads on airfoil
  const generateHighSpeedAeroLoads = async () => {
    addLog('🌊 Generating high-speed aerodynamic loads...', 'info');

    const velocity_ms = config.velocity / 3.6; // Convert km/h to m/s
    const rho = 1.225; // kg/m³
    const dynamicPressure = 0.5 * rho * velocity_ms * velocity_ms;

    // Simulate pressure distribution along chord
    const chordPoints = 100;
    const pressureDistribution = [];

    for (let i = 0; i < chordPoints; i++) {
      const x = i / (chordPoints - 1); // Normalized chord position [0,1]
      
      // Pressure coefficient based on thin airfoil theory + corrections
      const alpha_rad = (config.yawAngle * Math.PI) / 180;
      
      // Upper surface (suction side)
      const cp_upper = -2 * alpha_rad * Math.sqrt((1 - x) / x) - 0.5;
      
      // Lower surface (pressure side)
      const cp_lower = 2 * alpha_rad * Math.sqrt((1 - x) / x) + 0.2;
      
      // Convert to actual pressure
      const p_upper = -cp_upper * dynamicPressure;
      const p_lower = -cp_lower * dynamicPressure;
      
      pressureDistribution.push({
        x,
        cp_upper,
        cp_lower,
        pressure_upper: p_upper,
        pressure_lower: p_lower,
        normal_force: (p_lower - p_upper) * dynamicPressure,
      });
    }

    // Calculate integrated loads
    const cl = 2 * Math.PI * alpha_rad; // Thin airfoil theory
    const cd = 0.01 + 0.05 * alpha_rad * alpha_rad; // Induced + profile drag
    
    const chord = 0.5; // m
    const span = 1.8; // m
    const area = chord * span;
    
    const lift = cl * dynamicPressure * area;
    const drag = cd * dynamicPressure * area;

    addLog(`✅ Aerodynamic loads computed: L=${lift.toFixed(1)}N, D=${drag.toFixed(1)}N`, 'success', {
      lift, drag, cl, cd, dynamicPressure
    });

    return {
      pressureDistribution,
      lift,
      drag,
      cl,
      cd,
      dynamicPressure,
    };
  };

  // Perform modal analysis
  const performModalAnalysis = async () => {
    addLog('〰️ Starting modal analysis...', 'info');

    const { thickness, stiffenerCount, material } = config.structuralConfig;

    // Material properties
    const materials = {
      carbon_fiber: { E: 150e9, rho: 1600, nu: 0.3 },
      aluminum: { E: 70e9, rho: 2700, nu: 0.33 },
    };

    const mat = materials[material];
    
    // Simplified beam properties
    const L = 0.9; // Half-span (m)
    const h = thickness / 1000; // Convert mm to m
    const b = 0.1; // Width (m)
    const I = (b * h * h * h) / 12; // Second moment of area
    const m = mat.rho * b * h; // Mass per unit length

    // Calculate first 5 natural frequencies (Euler-Bernoulli beam)
    const frequencies = [];
    const dampingRatios = [];
    const modeShapes = [];

    for (let n = 1; n <= 5; n++) {
      const lambda_n = [(1.875, 4.694, 7.855, 10.996, 14.137)][n-1] || (2*n - 1) * Math.PI / 2;
      const omega_n = lambda_n * lambda_n * Math.sqrt((mat.E * I) / (m * L * L * L * L));
      const f_n = omega_n / (2 * Math.PI);
      
      // Damping increases with stiffener count
      const zeta = 0.015 + (stiffenerCount * 0.005);
      
      frequencies.push(f_n);
      dampingRatios.push(zeta);
      
      // Generate mode shape (simplified)
      const modeShape = [];
      for (let x = 0; x <= 1; x += 0.05) {
        const shape = Math.sin(lambda_n * x);
        modeShape.push({ x, amplitude: shape });
      }
      modeShapes.push(modeShape);

      addLog(`Mode ${n}: f=${f_n.toFixed(2)} Hz, ζ=${(zeta*100).toFixed(2)}%`, 'info');
    }

    return { frequencies, dampingRatios, modeShapes };
  };

  // Compute flutter speed using k-method
  const computeFlutterSpeed = async (modalData, aeroLoads) => {
    addLog('🎯 Computing flutter speed...', 'info');

    const { frequencies, dampingRatios } = modalData;
    
    // Simplified flutter analysis using k-method
    // Flutter occurs when aerodynamic damping overcomes structural damping
    
    const velocityRange = [];
    const vgData = [];
    
    for (let v = 50; v <= 400; v += 10) {
      const v_ms = v / 3.6;
      const q = 0.5 * 1.225 * v_ms * v_ms; // Dynamic pressure
      
      // For each mode, compute total damping
      const modeDampings = frequencies.map((f, idx) => {
        const omega = 2 * Math.PI * f;
        const k = omega * 0.5 / v_ms; // Reduced frequency
        
        // Aerodynamic damping (Theodorsen function approximation)
        const C_k = 1 / (1 + 0.5 * k * k); // Simplified
        const aero_damping = -0.5 * q * 0.5 * 1.8 * C_k / (omega * m_modal);
        
        // Total damping
        const total_damping = dampingRatios[idx] + aero_damping / (2 * omega);
        
        return total_damping;
      });
      
      // Store V-g data for plotting
      modeDampings.forEach((g, modeIdx) => {
        vgData.push({
          velocity: v,
          mode: modeIdx + 1,
          damping: g * 100, // Convert to percentage
        });
      });
      
      velocityRange.push({ velocity: v, dampings: modeDampings });
    }

    // Find flutter speed (first velocity where damping becomes negative)
    let flutterSpeed = 400; // Default high value
    let criticalMode = null;
    
    for (const point of velocityRange) {
      for (let i = 0; i < point.dampings.length; i++) {
        if (point.dampings[i] < 0) {
          flutterSpeed = point.velocity;
          criticalMode = i + 1;
          break;
        }
      }
      if (flutterSpeed < 400) break;
    }

    const flutterMargin = flutterSpeed / config.velocity;
    
    const status = flutterMargin > 1.5 ? 'safe' : (flutterMargin > 1.2 ? 'warning' : 'critical');
    const emoji = status === 'safe' ? '✅' : (status === 'warning' ? '⚠️' : '❌');
    
    addLog(`${emoji} Flutter speed: ${flutterSpeed.toFixed(1)} km/h (Margin: ${flutterMargin.toFixed(2)}x)`, 
           status === 'safe' ? 'success' : 'warning', 
           { flutterSpeed, flutterMargin, criticalMode });

    return {
      flutterSpeed,
      flutterMargin,
      criticalMode,
      vgData,
    };
  };

  // Main analysis workflow
  const runAeroelasticAnalysis = async () => {
    setIsAnalyzing(true);
    addLog('🚀 Starting comprehensive aeroelastic analysis...', 'success');

    try {
      // Step 1: Generate aerodynamic loads
      const aeroLoads = await generateHighSpeedAeroLoads();
      
      // Step 2: Perform modal analysis
      const modalData = await performModalAnalysis();
      
      // Step 3: Compute flutter characteristics
      const flutterData = await computeFlutterSpeed(modalData, aeroLoads);
      
      // Step 4: Estimate deformations
      const maxDisplacement = (aeroLoads.lift * 0.9 * 0.9 * 0.9) / (3 * 150e9 * (0.1 * 0.002 * 0.002 * 0.002) / 12);
      const tipDeflection = maxDisplacement * 1000; // Convert to mm
      const twistAngle = 0.5; // degrees (simplified)
      
      addLog(`📏 Max deformation: ${tipDeflection.toFixed(2)} mm, Twist: ${twistAngle.toFixed(2)}°`, 'info');
      
      // Update state
      const results = {
        modalData,
        flutterData,
        aeroLoads: {
          staticLoads: aeroLoads.pressureDistribution,
          dynamicLoads: [],
          pressureDistribution: aeroLoads.pressureDistribution,
        },
        deformation: {
          maxDisplacement,
          tipDeflection,
          twistAngle,
        },
      };
      
      setAnalysisResults(results);
      
      // Share modal data with other components
      if (onModalDataGenerated) {
        onModalDataGenerated({
          component: config.component,
          material: config.structuralConfig.material,
          thickness: config.structuralConfig.thickness,
          modalData: modalData,
          flutterData: flutterData,
          timestamp: new Date().toISOString(),
        });
      }
      
      addLog('✅ Aeroelastic analysis completed successfully!', 'success');
      
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
      results: analysisResults,
      timestamp: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `aeroelastic_analysis_${Date.now()}.json`;
    link.click();
    
    addLog('💾 Results exported successfully', 'success');
  };

  // Modal mass (simplified)
  const m_modal = 10; // kg

  return (
    <div className="aeroelastic-analysis-panel">
      <div className="panel-header">
        <h2>〰️ Aeroelastic Analysis Panel</h2>
        <p>Flutter Analysis • Modal Dynamics • High-Speed Aerodynamic Loads</p>
      </div>

      {/* Configuration Section */}
      <div className="config-section">
        <h3>Configuration</h3>
        
        <div className="config-grid">
          {/* Component Selection */}
          <div className="config-group">
            <label>Component</label>
            <select value={config.component} onChange={(e) => setConfig({...config, component: e.target.value})}>
              <option value="front_wing">Front Wing</option>
              <option value="rear_wing">Rear Wing</option>
              <option value="floor">Floor</option>
              <option value="diffuser">Diffuser</option>
            </select>
          </div>

          {/* NACA Profile */}
          <div className="config-group">
            <label>NACA Profile</label>
            <select value={config.nacaProfile} onChange={(e) => setConfig({...config, nacaProfile: e.target.value})}>
              <option value="NACA6412">NACA 6412</option>
              <option value="NACA4415">NACA 4415</option>
              <option value="NACA4418">NACA 4418</option>
              <option value="NACA9618">NACA 9618</option>
            </select>
          </div>

          {/* Velocity */}
          <div className="config-group">
            <label>Velocity: {config.velocity} km/h</label>
            <input
              type="range"
              min="50"
              max="400"
              value={config.velocity}
              onChange={(e) => setConfig({...config, velocity: parseInt(e.target.value)})}
            />
          </div>

          {/* Yaw Angle */}
          <div className="config-group">
            <label>Yaw Angle: {config.yawAngle}°</label>
            <input
              type="range"
              min="0"
              max="10"
              step="0.5"
              value={config.yawAngle}
              onChange={(e) => setConfig({...config, yawAngle: parseFloat(e.target.value)})}
            />
          </div>

          {/* Thickness */}
          <div className="config-group">
            <label>Thickness: {config.structuralConfig.thickness} mm</label>
            <input
              type="range"
              min="1.0"
              max="3.0"
              step="0.5"
              value={config.structuralConfig.thickness}
              onChange={(e) => setConfig({
                ...config,
                structuralConfig: {...config.structuralConfig, thickness: parseFloat(e.target.value)}
              })}
            />
          </div>

          {/* Stiffener Count */}
          <div className="config-group">
            <label>Stiffeners: {config.structuralConfig.stiffenerCount}</label>
            <input
              type="range"
              min="0"
              max="8"
              step="2"
              value={config.structuralConfig.stiffenerCount}
              onChange={(e) => setConfig({
                ...config,
                structuralConfig: {...config.structuralConfig, stiffenerCount: parseInt(e.target.value)}
              })}
            />
          </div>
        </div>

        <div className="action-buttons">
          <button 
            className="btn-analyze" 
            onClick={runAeroelasticAnalysis}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? '⏳ Analyzing...' : '▶️ Run Analysis'}
          </button>
          <button className="btn-export" onClick={exportResults}>
            💾 Export Results
          </button>
        </div>
      </div>

      {/* Results Grid */}
      <div className="results-grid">
        {/* Flutter Analysis */}
        <div className="result-panel flutter-panel">
          <h3>🎯 Flutter Analysis</h3>
          
          <div className="metrics-row">
            <div className="metric-card">
              <div className="metric-label">Flutter Speed</div>
              <div className={`metric-value ${
                analysisResults.flutterData.flutterMargin > 1.5 ? 'safe' : 
                analysisResults.flutterData.flutterMargin > 1.2 ? 'warning' : 'critical'
              }`}>
                {analysisResults.flutterData.flutterSpeed.toFixed(1)}
              </div>
              <div className="metric-unit">km/h</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Safety Margin</div>
              <div className={`metric-value ${
                analysisResults.flutterData.flutterMargin > 1.5 ? 'safe' : 
                analysisResults.flutterData.flutterMargin > 1.2 ? 'warning' : 'critical'
              }`}>
                {analysisResults.flutterData.flutterMargin.toFixed(2)}
              </div>
              <div className="metric-unit">×</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Critical Mode</div>
              <div className="metric-value">
                {analysisResults.flutterData.criticalMode || 'N/A'}
              </div>
              <div className="metric-unit">mode</div>
            </div>
          </div>

          {/* V-g Diagram */}
          {analysisResults.flutterData.vgData.length > 0 && (
            <div className="chart-container">
              <h4>V-g Diagram (Velocity-Damping)</h4>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={analysisResults.flutterData.vgData.filter(d => d.mode === 1)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="velocity" stroke="#808080" label={{ value: 'Velocity (km/h)', position: 'bottom' }} />
                  <YAxis stroke="#808080" label={{ value: 'Damping (%)', angle: -90, position: 'left' }} />
                  <Tooltip contentStyle={{background: 'rgba(0,0,0,0.8)', border: '1px solid #00c8ff'}} />
                  <ReferenceLine y={0} stroke="#ff0000" strokeDasharray="5 5" label="Flutter Boundary" />
                  <ReferenceArea y1={0} y2={100} fill="#00ff88" fillOpacity={0.1} />
                  <ReferenceArea y1={-100} y2={0} fill="#ff0000" fillOpacity={0.1} />
                  <Line type="monotone" dataKey="damping" stroke="#00c8ff" strokeWidth={2} dot={false} name="Mode 1" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Modal Analysis */}
        <div className="result-panel modal-panel">
          <h3>〰️ Modal Analysis</h3>
          
          {analysisResults.modalData.frequencies.length > 0 && (
            <>
              <div className="mode-list">
                {analysisResults.modalData.frequencies.map((freq, idx) => (
                  <div 
                    key={idx} 
                    className={`mode-item ${selectedMode === idx ? 'selected' : ''}`}
                    onClick={() => setSelectedMode(idx)}
                  >
                    <span className="mode-number">Mode {idx + 1}</span>
                    <span className="mode-frequency">{freq.toFixed(2)} Hz</span>
                    <span className="mode-damping">ζ = {(analysisResults.modalData.dampingRatios[idx] * 100).toFixed(2)}%</span>
                  </div>
                ))}
              </div>

              {/* Mode Shape Visualization */}
              <div className="chart-container">
                <h4>Mode Shape - Mode {selectedMode + 1}</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={analysisResults.modalData.modeShapes[selectedMode]}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="x" stroke="#808080" label={{ value: 'Span Position', position: 'bottom' }} />
                    <YAxis stroke="#808080" label={{ value: 'Amplitude', angle: -90, position: 'left' }} />
                    <Tooltip contentStyle={{background: 'rgba(0,0,0,0.8)', border: '1px solid #00ff88'}} />
                    <Area type="monotone" dataKey="amplitude" stroke="#00ff88" fill="#00ff88" fillOpacity={0.3} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
        </div>

        {/* Aerodynamic Loads */}
        <div className="result-panel aero-panel">
          <h3>🌊 Aerodynamic Loads</h3>
          
          {analysisResults.aeroLoads.pressureDistribution.length > 0 && (
            <div className="chart-container">
              <h4>Pressure Distribution</h4>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={analysisResults.aeroLoads.pressureDistribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="x" stroke="#808080" label={{ value: 'Chord Position', position: 'bottom' }} />
                  <YAxis stroke="#808080" label={{ value: 'Cp', angle: -90, position: 'left' }} />
                  <Tooltip contentStyle={{background: 'rgba(0,0,0,0.8)', border: '1px solid #00c8ff'}} />
                  <Line type="monotone" dataKey="cp_upper" stroke="#00c8ff" strokeWidth={2} dot={false} name="Upper Surface" />
                  <Line type="monotone" dataKey="cp_lower" stroke="#ff8800" strokeWidth={2} dot={false} name="Lower Surface" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="load-summary">
            <div className="load-item">
              <span>Dynamic Pressure:</span>
              <span>{(analysisResults.aeroLoads.pressureDistribution[0]?.normal_force || 0).toFixed(1)} Pa</span>
            </div>
          </div>
        </div>

        {/* Deformation */}
        <div className="result-panel deformation-panel">
          <h3>📏 Structural Deformation</h3>
          
          <div className="deformation-metrics">
            <div className="deform-card">
              <div className="deform-label">Tip Deflection</div>
              <div className="deform-value">{analysisResults.deformation.tipDeflection.toFixed(2)}</div>
              <div className="deform-unit">mm</div>
            </div>

            <div className="deform-card">
              <div className="deform-label">Twist Angle</div>
              <div className="deform-value">{analysisResults.deformation.twistAngle.toFixed(2)}</div>
              <div className="deform-unit">degrees</div>
            </div>

            <div className="deform-card">
              <div className="deform-label">Max Displacement</div>
              <div className="deform-value">{(analysisResults.deformation.maxDisplacement * 1000).toFixed(2)}</div>
              <div className="deform-unit">mm</div>
            </div>
          </div>
        </div>
      </div>

      {/* Logs Panel */}
      <div className="logs-panel">
        <h3>📝 Analysis Logs</h3>
        <div className="logs-container">
          {logs.map((log, idx) => (
            <div key={idx} className={`log-entry log-${log.type}`}>
              <span className="log-time">[{log.timestamp}]</span>
              <span className="log-message">{log.message}</span>
              {log.data && (
                <span className="log-data">{JSON.stringify(log.data)}</span>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AeroelasticAnalysisPanel;
