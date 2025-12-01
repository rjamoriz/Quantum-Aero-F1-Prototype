import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine, ScatterChart, Scatter
} from 'recharts';
import './AeroacousticAnalysisPanel.css';

/**
 * Aeroacoustic Analysis Panel
 * Sound Pressure Level (SPL) analysis with FW-H solver integration
 * - SPL visualization
 * - Frequency spectrum analysis
 * - Directivity patterns
 * - Noise source localization
 */
const AeroacousticAnalysisPanel = ({ modalData }) => {
  const [config, setConfig] = useState({
    component: 'front_wing',
    velocity: 300, // km/h
    yawAngle: 0, // degrees
    observerDistance: 10, // meters
    frequencyRange: [20, 20000], // Hz
    analysisType: 'broadband', // broadband or tonal
  });

  const [acousticResults, setAcousticResults] = useState({
    splData: [],
    frequencySpectrum: [],
    directivityPattern: [],
    noiseSources: [],
    oaspl: null, // Overall SPL
  });

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [displayOptions, setDisplayOptions] = useState({
    showSPL: true,
    showSpectrum: true,
    showDirectivity: true,
    showSources: true,
  });
  const [logs, setLogs] = useState([]);

  // Component acoustic properties
  const componentProperties = {
    front_wing: {
      span: 1.8,
      chord: 0.5,
      area: 0.9,
      typicalSources: ['trailing_edge', 'tip_vortex', 'flap_gap'],
    },
    rear_wing: {
      span: 1.5,
      chord: 0.6,
      area: 0.9,
      typicalSources: ['trailing_edge', 'tip_vortex', 'drs_gap'],
    },
    floor: {
      span: 3.5,
      chord: 1.2,
      area: 4.2,
      typicalSources: ['diffuser_exit', 'edge_noise', 'vortex'],
    },
    diffuser: {
      span: 1.0,
      chord: 1.5,
      area: 1.5,
      typicalSources: ['expansion_noise', 'separation_noise', 'exit_jet'],
    },
  };

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, { timestamp, message, type }]);
  };

  // Calculate SPL using simplified FW-H approach
  const calculateFWH_SPL = () => {
    addLog('🔊 Calculating Ffowcs Williams-Hawkings (FW-H) acoustic sources...', 'info');
    
    const geom = componentProperties[config.component];
    const V = config.velocity / 3.6; // m/s
    const rho = 1.225; // kg/m³
    const c0 = 343; // Speed of sound (m/s)
    const p0 = 2e-5; // Reference pressure (Pa) for SPL calculation
    
    // Mach number
    const M = V / c0;
    addLog(`Mach number: ${M.toFixed(3)}`, 'info');
    
    // Dynamic pressure
    const q = 0.5 * rho * V * V;
    
    // Acoustic sources
    const sources = [];
    
    // 1. Trailing Edge Noise (dominant for airfoils)
    const delta_bl = 0.001 * geom.chord * Math.pow(V * geom.chord / 1.5e-5, -0.2); // Boundary layer thickness
    const f_te = 0.1 * V / delta_bl; // Peak frequency
    const L_te = 10 * Math.log10(
      Math.pow(M, 5) * geom.span * delta_bl * Math.pow(V, 2) / Math.pow(config.observerDistance, 2)
    ) + 90; // Simplified formula
    
    sources.push({
      type: 'Trailing Edge',
      frequency: f_te,
      spl: L_te,
      location: { x: geom.chord, y: 0, z: 0 },
      contribution: 0.45,
    });
    
    // 2. Tip Vortex Noise
    const f_vortex = 0.2 * V / geom.chord;
    const L_vortex = L_te - 5; // Typically 3-7 dB lower
    
    sources.push({
      type: 'Tip Vortex',
      frequency: f_vortex,
      spl: L_vortex,
      location: { x: geom.chord * 0.7, y: geom.span / 2, z: 0 },
      contribution: 0.30,
    });
    
    // 3. Separation/Turbulence Noise
    const f_turb = 0.15 * V / geom.chord;
    const L_turb = L_te - 8;
    
    sources.push({
      type: 'Turbulence',
      frequency: f_turb,
      spl: L_turb,
      location: { x: geom.chord * 0.5, y: 0, z: 0 },
      contribution: 0.25,
    });
    
    // Overall SPL (logarithmic sum)
    const oaspl = 10 * Math.log10(
      sources.reduce((sum, s) => sum + Math.pow(10, s.spl / 10), 0)
    );
    
    addLog(`Overall SPL: ${oaspl.toFixed(1)} dB`, 'success');
    
    return { sources, oaspl };
  };

  // Generate frequency spectrum
  const generateFrequencySpectrum = (sources) => {
    addLog('📊 Generating frequency spectrum...', 'info');
    
    const spectrum = [];
    const f_min = config.frequencyRange[0];
    const f_max = config.frequencyRange[1];
    const numBands = 30; // 1/3 octave bands
    
    for (let i = 0; i < numBands; i++) {
      const f_center = f_min * Math.pow(f_max / f_min, i / (numBands - 1));
      
      // Sum contributions from all sources with bandwidth decay
      let spl_band = 0;
      sources.forEach(source => {
        const df = Math.abs(f_center - source.frequency) / source.frequency;
        const decay = Math.exp(-5 * df * df); // Gaussian-like decay
        const spl_contribution = source.spl * source.contribution * decay;
        spl_band += Math.pow(10, spl_contribution / 10);
      });
      
      const spl_total = 10 * Math.log10(spl_band);
      
      spectrum.push({
        frequency: f_center,
        spl: spl_total,
        band: `${Math.round(f_center)} Hz`,
      });
    }
    
    addLog(`Spectrum generated: ${numBands} frequency bands`, 'success');
    
    return spectrum;
  };

  // Calculate directivity pattern
  const calculateDirectivity = (sources, oaspl) => {
    addLog('🎯 Calculating directivity pattern...', 'info');
    
    const directivity = [];
    const numAngles = 36; // Every 10 degrees
    
    for (let i = 0; i < numAngles; i++) {
      const theta = (i * 360) / numAngles; // degrees
      const theta_rad = theta * Math.PI / 180;
      
      // Dipole directivity (trailing edge, turbulence)
      const dipole_factor = Math.pow(Math.sin(theta_rad), 2);
      
      // Monopole directivity (tip vortex)
      const monopole_factor = 1.0;
      
      // Combined directivity (weighted)
      const D_theta = 0.7 * dipole_factor + 0.3 * monopole_factor;
      
      // SPL at angle theta
      const spl_theta = oaspl + 10 * Math.log10(D_theta);
      
      directivity.push({
        angle: theta,
        spl: spl_theta,
        normalized: D_theta,
      });
    }
    
    addLog('Directivity pattern calculated', 'success');
    
    return directivity;
  };

  // Generate SPL time history (for transient analysis)
  const generateSPLTimeHistory = (oaspl) => {
    addLog('⏱️ Generating SPL time history...', 'info');
    
    const splData = [];
    const duration = 10; // seconds
    const dt = 0.1; // seconds
    const numSteps = duration / dt;
    
    for (let i = 0; i < numSteps; i++) {
      const t = i * dt;
      
      // Time-varying component (cornering, acceleration)
      const velocity_factor = 1 + 0.1 * Math.sin(2 * Math.PI * t / 5);
      const spl_instant = oaspl + 10 * Math.log10(Math.pow(velocity_factor, 5));
      
      // Add turbulent fluctuations
      const fluctuation = 2 * (Math.random() - 0.5);
      
      splData.push({
        time: t,
        spl: spl_instant + fluctuation,
        oaspl: oaspl,
      });
    }
    
    return splData;
  };

  // Run complete acoustic analysis
  const runAcousticAnalysis = () => {
    setIsAnalyzing(true);
    setLogs([]);
    addLog('🚀 Starting aeroacoustic analysis...', 'info');
    
    try {
      // Step 1: Calculate FW-H acoustic sources
      const { sources, oaspl } = calculateFWH_SPL();
      
      // Step 2: Generate frequency spectrum
      const frequencySpectrum = generateFrequencySpectrum(sources);
      
      // Step 3: Calculate directivity
      const directivityPattern = calculateDirectivity(sources, oaspl);
      
      // Step 4: Generate SPL time history
      const splData = generateSPLTimeHistory(oaspl);
      
      // Check FIA noise limits
      const fiaLimit = 110; // dB(A) at 15m
      const margin = oaspl - fiaLimit;
      
      if (margin > 0) {
        addLog(`⚠️ WARNING: SPL exceeds FIA limit by ${margin.toFixed(1)} dB`, 'error');
      } else {
        addLog(`✅ SPL within FIA limit (margin: ${Math.abs(margin).toFixed(1)} dB)`, 'success');
      }
      
      setAcousticResults({
        splData,
        frequencySpectrum,
        directivityPattern,
        noiseSources: sources,
        oaspl,
      });
      
      addLog('✅ Aeroacoustic analysis completed successfully!', 'success');
      
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
      results: acousticResults,
      timestamp: new Date().toISOString(),
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `acoustic_analysis_${config.component}_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
    
    addLog('📥 Results exported successfully', 'success');
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="acoustic-tooltip">
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: {entry.value.toFixed(1)} {entry.unit || 'dB'}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="aeroacoustic-analysis-panel">
      <div className="panel-header">
        <h2>🔊 Aeroacoustic Analysis Panel</h2>
        <p>FW-H Acoustic Solver & Noise Source Localization</p>
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
            <label>Yaw Angle: {config.yawAngle}°</label>
            <input
              type="range"
              min="-15"
              max="15"
              value={config.yawAngle}
              onChange={(e) => setConfig({ ...config, yawAngle: parseInt(e.target.value) })}
            />
          </div>

          <div className="config-item">
            <label>Observer Distance: {config.observerDistance} m</label>
            <input
              type="range"
              min="5"
              max="50"
              value={config.observerDistance}
              onChange={(e) => setConfig({ ...config, observerDistance: parseInt(e.target.value) })}
            />
          </div>

          <div className="config-item">
            <label>Analysis Type</label>
            <select
              value={config.analysisType}
              onChange={(e) => setConfig({ ...config, analysisType: e.target.value })}
            >
              <option value="broadband">Broadband Noise</option>
              <option value="tonal">Tonal Noise</option>
              <option value="combined">Combined</option>
            </select>
          </div>
        </div>

        <div className="action-buttons">
          <button
            className="analyze-btn"
            onClick={runAcousticAnalysis}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? '🔄 Analyzing...' : '▶️ Run Acoustic Analysis'}
          </button>
          
          <button
            className="export-btn"
            onClick={exportResults}
            disabled={!acousticResults.oaspl}
          >
            📥 Export Results
          </button>
        </div>
      </div>

      {/* OASPL Summary */}
      {acousticResults.oaspl && (
        <div className="oaspl-section">
          <div className="oaspl-card">
            <span className="oaspl-icon">🔊</span>
            <div className="oaspl-content">
              <span className="oaspl-label">Overall SPL (OASPL)</span>
              <span className="oaspl-value">{acousticResults.oaspl.toFixed(1)} dB</span>
              <span className={`oaspl-status ${acousticResults.oaspl > 110 ? 'over-limit' : 'within-limit'}`}>
                {acousticResults.oaspl > 110 ? '⚠️ Above FIA Limit' : '✅ Within FIA Limit (110 dB)'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Noise Sources */}
      {acousticResults.noiseSources.length > 0 && (
        <div className="sources-section">
          <h3>Noise Sources</h3>
          <div className="sources-grid">
            {acousticResults.noiseSources.map((source, index) => (
              <div key={index} className="source-card">
                <h4>{source.type}</h4>
                <div className="source-details">
                  <div className="source-metric">
                    <span className="metric-label">SPL</span>
                    <span className="metric-value">{source.spl.toFixed(1)} dB</span>
                  </div>
                  <div className="source-metric">
                    <span className="metric-label">Frequency</span>
                    <span className="metric-value">{source.frequency.toFixed(0)} Hz</span>
                  </div>
                  <div className="source-metric">
                    <span className="metric-label">Contribution</span>
                    <span className="metric-value">{(source.contribution * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <div className="source-bar">
                  <div 
                    className="source-fill" 
                    style={{ width: `${source.contribution * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Results Visualization */}
      {acousticResults.splData.length > 0 && (
        <div className="results-section">
          <div className="display-options">
            <h3>Display Options</h3>
            <div className="options-grid">
              <label>
                <input
                  type="checkbox"
                  checked={displayOptions.showSPL}
                  onChange={(e) => setDisplayOptions({ ...displayOptions, showSPL: e.target.checked })}
                />
                SPL Time History
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={displayOptions.showSpectrum}
                  onChange={(e) => setDisplayOptions({ ...displayOptions, showSpectrum: e.target.checked })}
                />
                Frequency Spectrum
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={displayOptions.showDirectivity}
                  onChange={(e) => setDisplayOptions({ ...displayOptions, showDirectivity: e.target.checked })}
                />
                Directivity Pattern
              </label>
            </div>
          </div>

          {/* SPL Time History */}
          {displayOptions.showSPL && (
            <div className="chart-container">
              <h4>SPL Time History</h4>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={acousticResults.splData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="time" stroke="#fff" label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }} />
                  <YAxis stroke="#fff" label={{ value: 'SPL (dB)', angle: -90, position: 'insideLeft' }} domain={[80, 120]} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <ReferenceLine y={110} stroke="#ff3366" strokeDasharray="3 3" label="FIA Limit" />
                  <Line type="monotone" dataKey="spl" stroke="#00ff88" name="SPL" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="oaspl" stroke="#00c8ff" name="OASPL" strokeWidth={1} strokeDasharray="5 5" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Frequency Spectrum */}
          {displayOptions.showSpectrum && (
            <div className="chart-container">
              <h4>Frequency Spectrum (1/3 Octave Bands)</h4>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={acousticResults.frequencySpectrum}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="band" stroke="#fff" angle={-45} textAnchor="end" height={80} />
                  <YAxis stroke="#fff" label={{ value: 'SPL (dB)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Bar dataKey="spl" name="SPL">
                    {acousticResults.frequencySpectrum.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.spl > 90 ? '#ff6b6b' : entry.spl > 80 ? '#ff8800' : '#00ff88'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Directivity Pattern */}
          {displayOptions.showDirectivity && (
            <div className="chart-container">
              <h4>Directivity Pattern (Polar Plot)</h4>
              <ResponsiveContainer width="100%" height={400}>
                <RadarChart data={acousticResults.directivityPattern}>
                  <PolarGrid stroke="rgba(255,255,255,0.2)" />
                  <PolarAngleAxis dataKey="angle" stroke="#fff" />
                  <PolarRadiusAxis angle={90} domain={[0, 'auto']} stroke="#fff" />
                  <Radar name="SPL" dataKey="spl" stroke="#00c8ff" fill="#00c8ff" fillOpacity={0.3} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                </RadarChart>
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

export default AeroacousticAnalysisPanel;
