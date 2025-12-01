import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, ReferenceLine, ReferenceArea
} from 'recharts';
import './FlutterAnalysisPanel.css';

/**
 * Flutter Analysis Panel - Comprehensive Multi-Mode V-g Diagrams
 * Dedicated flutter stability visualization with parametric studies
 */
const FlutterAnalysisPanel = ({ modalData = null }) => {
  const [flutterAnalysis, setFlutterAnalysis] = useState({
    modes: [],
    criticalMode: null,
    flutterSpeed: 0,
    safetyMargin: 0,
    vgDiagrams: [],
    parametricStudy: [],
  });

  const [analysisConfig, setAnalysisConfig] = useState({
    component: 'front_wing',
    velocityRange: { min: 50, max: 450, step: 5 },
    airDensity: 1.225,
    thickness: 2.0,
    material: 'carbon_fiber',
    stiffenerCount: 4,
  });

  const [displayOptions, setDisplayOptions] = useState({
    showIndividualModes: true,
    showCombined: true,
    showParametric: false,
    showSafetyZones: true,
  });

  const [selectedModes, setSelectedModes] = useState([0, 1, 2, 3, 4]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [logs, setLogs] = useState([]);

  const materials = {
    carbon_fiber: { E: 150e9, rho: 1600, nu: 0.3, name: 'Carbon Fiber' },
    aluminum: { E: 70e9, rho: 2700, nu: 0.33, name: 'Aluminum 7075' },
    titanium: { E: 110e9, rho: 4500, nu: 0.34, name: 'Titanium Ti-6Al-4V' },
  };

  const componentGeometry = {
    front_wing: { span: 1.8, chord: 0.5, area: 0.9, aspectRatio: 3.6 },
    rear_wing: { span: 1.5, chord: 0.6, area: 0.9, aspectRatio: 2.5 },
    floor: { span: 3.5, chord: 1.2, area: 4.2, aspectRatio: 2.92 },
    diffuser: { span: 1.0, chord: 1.5, area: 1.5, aspectRatio: 0.67 },
  };

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [{ timestamp, message, type }, ...prev].slice(0, 100));
  };

  useEffect(() => {
    if (modalData) {
      performFlutterAnalysis();
    }
  }, [modalData, analysisConfig]);

  const performFlutterAnalysis = async () => {
    setIsAnalyzing(true);
    addLog('🔬 Starting comprehensive flutter analysis...', 'info');

    try {
      const geom = componentGeometry[analysisConfig.component];
      const mat = materials[analysisConfig.material];

      const modes = modalData ? modalData.frequencies.map((freq, idx) => ({
        index: idx,
        frequency: freq,
        damping: modalData.dampingRatios[idx],
        type: ['bending', 'torsion', 'bending', 'coupled', 'local'][idx % 5],
      })) : generateDefaultModes();

      addLog(`📊 Analyzing ${modes.length} structural modes`, 'info');

      const vgDiagrams = [];
      let globalFlutterSpeed = Infinity;
      let criticalModeIndex = -1;

      for (let modeIdx = 0; modeIdx < modes.length; modeIdx++) {
        if (!selectedModes.includes(modeIdx)) continue;

        const mode = modes[modeIdx];
        const vgData = computeVgDiagram(mode, geom, mat);
        
        vgDiagrams.push({
          modeIndex: modeIdx,
          modeName: `Mode ${modeIdx + 1} (${mode.type})`,
          frequency: mode.frequency,
          data: vgData.points,
          flutterSpeed: vgData.flutterSpeed,
          flutterDamping: vgData.flutterDamping,
        });

        if (vgData.flutterSpeed < globalFlutterSpeed) {
          globalFlutterSpeed = vgData.flutterSpeed;
          criticalModeIndex = modeIdx;
        }

        addLog(`  Mode ${modeIdx + 1}: V_f = ${vgData.flutterSpeed.toFixed(1)} km/h`, 
               vgData.flutterSpeed > 350 ? 'success' : 'warning');
      }

      const operatingSpeed = 350;
      const safetyMargin = globalFlutterSpeed / operatingSpeed;

      addLog('📈 Running parametric sensitivity analysis...', 'info');
      const parametricStudy = performParametricStudy(modes[criticalModeIndex], geom, mat);

      setFlutterAnalysis({
        modes,
        criticalMode: criticalModeIndex,
        flutterSpeed: globalFlutterSpeed,
        safetyMargin,
        vgDiagrams,
        parametricStudy,
      });

      const status = safetyMargin > 1.5 ? 'success' : safetyMargin > 1.2 ? 'warning' : 'error';
      addLog(`✅ Flutter Speed: ${globalFlutterSpeed.toFixed(1)} km/h (Margin: ${safetyMargin.toFixed(2)}x)`, status);

    } catch (error) {
      addLog(`❌ Analysis failed: ${error.message}`, 'error');
      console.error(error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const computeVgDiagram = (mode, geom, mat) => {
    const { min, max, step } = analysisConfig.velocityRange;
    const points = [];
    
    const L = geom.span;
    const c = geom.chord;
    const S = geom.area;
    const rho = analysisConfig.airDensity;
    
    const h = analysisConfig.thickness / 1000;
    const b = 0.1;
    const I = (b * h * h * h) / 12;
    const m_modal = mat.rho * b * h * L / 4;
    
    const omega = 2 * Math.PI * mode.frequency;
    const zeta = mode.damping;

    let flutterSpeed = max;
    let flutterDamping = zeta;

    for (let V_kmh = min; V_kmh <= max; V_kmh += step) {
      const V = V_kmh / 3.6;
      const k = (omega * c) / (2 * V);
      const C_k = 1 / (1 + 0.5 * k * k);
      const q = 0.5 * rho * V * V;
      
      let g_aero = 0;
      if (mode.type === 'bending') {
        g_aero = -(Math.PI * rho * V * c * c * S) / (4 * omega * m_modal) * C_k;
      } else if (mode.type === 'torsion') {
        g_aero = -(Math.PI * rho * V * c * c * c * S) / (8 * omega * m_modal) * C_k;
      } else {
        g_aero = -(Math.PI * rho * V * c * c * S) / (6 * omega * m_modal) * C_k;
      }

      const zeta_total = zeta + g_aero;

      points.push({
        velocity: V_kmh,
        damping: zeta_total * 100,
        dampingAbsolute: zeta_total,
        reducedFreq: k,
      });

      if (zeta_total < 0 && V_kmh < flutterSpeed) {
        flutterSpeed = V_kmh;
        flutterDamping = zeta_total;
      }
    }

    return {
      points,
      flutterSpeed: flutterSpeed === max ? Infinity : flutterSpeed,
      flutterDamping,
    };
  };

  const performParametricStudy = (mode, geom, mat) => {
    const thicknessRange = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    const results = [];

    thicknessRange.forEach(thickness => {
      const h = thickness / 1000;
      const L = geom.span;
      const c = geom.chord;
      const S = geom.area;
      const rho = analysisConfig.airDensity;
      const b = 0.1;
      const I = (b * h * h * h) / 12;
      
      const E = mat.E;
      const rho_mat = mat.rho;
      const m = rho_mat * b * h;
      
      const lambda1 = 1.875;
      const omega_n = lambda1 * lambda1 * Math.sqrt((E * I) / (m * L * L * L * L));
      const f_n = omega_n / (2 * Math.PI);
      
      const m_modal = m * L / 4;
      const omega = 2 * Math.PI * f_n;
      const zeta = mode.damping;

      let V_flutter = 450;
      for (let V_kmh = 50; V_kmh <= 450; V_kmh += 5) {
        const V = V_kmh / 3.6;
        const k = (omega * c) / (2 * V);
        const C_k = 1 / (1 + 0.5 * k * k);
        
        const g_aero = -(Math.PI * rho * V * c * c * S) / (4 * omega * m_modal) * C_k;
        const zeta_total = zeta + g_aero;

        if (zeta_total < 0) {
          V_flutter = V_kmh;
          break;
        }
      }

      results.push({
        thickness,
        frequency: f_n,
        flutterSpeed: V_flutter,
        margin: V_flutter / 350,
      });
    });

    return results;
  };

  const generateDefaultModes = () => {
    return [
      { index: 0, frequency: 18.5, damping: 0.025, type: 'bending' },
      { index: 1, frequency: 32.4, damping: 0.020, type: 'torsion' },
      { index: 2, frequency: 47.2, damping: 0.028, type: 'bending' },
      { index: 3, frequency: 65.8, damping: 0.022, type: 'coupled' },
      { index: 4, frequency: 89.3, damping: 0.030, type: 'local' },
    ];
  };

  const toggleMode = (modeIndex) => {
    if (selectedModes.includes(modeIndex)) {
      setSelectedModes(selectedModes.filter(m => m !== modeIndex));
    } else {
      setSelectedModes([...selectedModes, modeIndex].sort());
    }
  };

  const exportResults = () => {
    const data = {
      config: analysisConfig,
      analysis: flutterAnalysis,
      timestamp: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `flutter_analysis_${Date.now()}.json`;
    link.click();
    
    addLog('💾 Results exported successfully', 'success');
  };

  const exportVgCSV = () => {
    let csv = 'Mode,Velocity (km/h),Damping (%),Reduced Frequency\n';
    
    flutterAnalysis.vgDiagrams.forEach(diagram => {
      diagram.data.forEach(point => {
        csv += `${diagram.modeName},${point.velocity},${point.damping},${point.reducedFreq}\n`;
      });
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `vg_diagrams_${Date.now()}.csv`;
    link.click();
    
    addLog('📊 V-g data exported as CSV', 'success');
  };

  const VgTooltip = ({ active, payload, label }) => {
    if (!active || !payload || !payload.length) return null;

    return (
      <div className="vg-tooltip">
        <p className="tooltip-velocity">V = {label} km/h</p>
        {payload.map((entry, index) => (
          <p key={index} style={{ color: entry.color }}>
            {entry.name}: {entry.value.toFixed(3)}%
          </p>
        ))}
      </div>
    );
  };

  return (
    <div className="flutter-analysis-panel">
      <div className="panel-header">
        <h2>🦋 Flutter Analysis Panel</h2>
        <p>Comprehensive V-g Diagram Visualization • Critical Speed Identification • Parameter Sensitivity</p>
      </div>

      <div className="config-section">
        <h3>Configuration</h3>
        <div className="config-grid">
          <div className="config-group">
            <label>Component</label>
            <select 
              value={analysisConfig.component} 
              onChange={(e) => setAnalysisConfig({...analysisConfig, component: e.target.value})}
            >
              <option value="front_wing">Front Wing</option>
              <option value="rear_wing">Rear Wing</option>
              <option value="floor">Floor</option>
              <option value="diffuser">Diffuser</option>
            </select>
          </div>

          <div className="config-group">
            <label>Material</label>
            <select 
              value={analysisConfig.material} 
              onChange={(e) => setAnalysisConfig({...analysisConfig, material: e.target.value})}
            >
              <option value="carbon_fiber">Carbon Fiber</option>
              <option value="aluminum">Aluminum 7075</option>
              <option value="titanium">Titanium Ti-6Al-4V</option>
            </select>
          </div>

          <div className="config-group">
            <label>Thickness: {analysisConfig.thickness} mm</label>
            <input
              type="range"
              min="1"
              max="5"
              step="0.5"
              value={analysisConfig.thickness}
              onChange={(e) => setAnalysisConfig({...analysisConfig, thickness: parseFloat(e.target.value)})}
            />
          </div>

          <div className="config-group">
            <label>Stiffeners: {analysisConfig.stiffenerCount}</label>
            <input
              type="range"
              min="0"
              max="10"
              step="1"
              value={analysisConfig.stiffenerCount}
              onChange={(e) => setAnalysisConfig({...analysisConfig, stiffenerCount: parseInt(e.target.value)})}
            />
          </div>
        </div>

        <div className="action-buttons">
          <button 
            className="btn-analyze" 
            onClick={performFlutterAnalysis}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? '⏳ Analyzing...' : '▶️ Run Analysis'}
          </button>
          <button className="btn-export" onClick={exportResults}>
            💾 Export JSON
          </button>
          <button className="btn-export" onClick={exportVgCSV}>
            📊 Export CSV
          </button>
        </div>
      </div>

      {flutterAnalysis.flutterSpeed > 0 && (
        <div className="summary-section">
          <div className="metric-card critical">
            <div className="metric-label">Critical Flutter Speed</div>
            <div className="metric-value">{flutterAnalysis.flutterSpeed.toFixed(1)} km/h</div>
            <div className="metric-subtitle">
              Mode {flutterAnalysis.criticalMode + 1} 
              {flutterAnalysis.modes[flutterAnalysis.criticalMode] && 
                ` (${flutterAnalysis.modes[flutterAnalysis.criticalMode].type})`}
            </div>
          </div>

          <div className={`metric-card ${flutterAnalysis.safetyMargin > 1.5 ? 'safe' : flutterAnalysis.safetyMargin > 1.2 ? 'warning' : 'critical'}`}>
            <div className="metric-label">Safety Margin</div>
            <div className="metric-value">{flutterAnalysis.safetyMargin.toFixed(2)}x</div>
            <div className="metric-subtitle">
              {flutterAnalysis.safetyMargin > 1.5 ? '✅ Safe' : 
               flutterAnalysis.safetyMargin > 1.2 ? '⚠️ Acceptable' : '❌ Critical'}
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-label">Operating Speed</div>
            <div className="metric-value">350 km/h</div>
            <div className="metric-subtitle">Max F1 Speed</div>
          </div>

          <div className="metric-card">
            <div className="metric-label">Active Modes</div>
            <div className="metric-value">{selectedModes.length}</div>
            <div className="metric-subtitle">of {flutterAnalysis.modes.length} total</div>
          </div>
        </div>
      )}

      <div className="mode-selection-section">
        <h3>Mode Selection</h3>
        <div className="mode-toggles">
          {flutterAnalysis.modes.map((mode, idx) => (
            <button
              key={idx}
              className={`mode-toggle ${selectedModes.includes(idx) ? 'active' : ''}`}
              onClick={() => toggleMode(idx)}
            >
              <span className="mode-number">Mode {idx + 1}</span>
              <span className="mode-type">{mode.type}</span>
              <span className="mode-freq">{mode.frequency.toFixed(1)} Hz</span>
            </button>
          ))}
        </div>
      </div>

      {displayOptions.showIndividualModes && flutterAnalysis.vgDiagrams.length > 0 && (
        <div className="vg-diagrams-section">
          <h3>Individual V-g Diagrams</h3>
          <div className="vg-grid">
            {flutterAnalysis.vgDiagrams.map((diagram, idx) => (
              <div key={idx} className="vg-diagram-card">
                <h4>
                  {diagram.modeName}
                  {diagram.modeIndex === flutterAnalysis.criticalMode && (
                    <span className="critical-badge">CRITICAL</span>
                  )}
                </h4>
                <div className="diagram-info">
                  <span>f = {diagram.frequency.toFixed(2)} Hz</span>
                  <span className={diagram.flutterSpeed < 350 ? 'flutter-warning' : 'flutter-safe'}>
                    V_f = {diagram.flutterSpeed === Infinity ? '∞' : diagram.flutterSpeed.toFixed(1)} km/h
                  </span>
                </div>
                
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={diagram.data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis 
                      dataKey="velocity" 
                      stroke="#808080" 
                      label={{ value: 'Velocity (km/h)', position: 'bottom', offset: -5 }}
                    />
                    <YAxis 
                      stroke="#808080" 
                      label={{ value: 'Damping (%)', angle: -90, position: 'left' }}
                    />
                    <Tooltip content={<VgTooltip />} />
                    <ReferenceLine y={0} stroke="#ff3366" strokeWidth={2} strokeDasharray="5 5" />
                    <ReferenceLine x={350} stroke="#ff8800" strokeWidth={1} strokeDasharray="3 3" label="V_max" />
                    
                    {displayOptions.showSafetyZones && (
                      <>
                        <ReferenceArea y1={0} y2={10} fill="#00ff88" fillOpacity={0.1} />
                        <ReferenceArea y1={-10} y2={0} fill="#ff3366" fillOpacity={0.1} />
                      </>
                    )}
                    
                    <Line 
                      type="monotone" 
                      dataKey="damping" 
                      stroke={diagram.modeIndex === flutterAnalysis.criticalMode ? '#ff3366' : '#00c8ff'}
                      strokeWidth={2}
                      dot={false}
                      name="Total Damping"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        </div>
      )}

      {displayOptions.showCombined && flutterAnalysis.vgDiagrams.length > 0 && (
        <div className="combined-vg-section">
          <h3>Combined V-g Diagram (All Modes)</h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis 
                type="number"
                domain={[analysisConfig.velocityRange.min, analysisConfig.velocityRange.max]}
                stroke="#808080" 
                label={{ value: 'Velocity (km/h)', position: 'bottom' }}
              />
              <YAxis 
                stroke="#808080" 
                domain={[-5, 5]}
                label={{ value: 'Damping (%)', angle: -90, position: 'left' }}
              />
              <Tooltip content={<VgTooltip />} />
              <Legend />
              <ReferenceLine y={0} stroke="#ff3366" strokeWidth={3} strokeDasharray="5 5" />
              <ReferenceLine x={350} stroke="#ff8800" strokeWidth={2} strokeDasharray="3 3" label="V_max" />
              
              {displayOptions.showSafetyZones && (
                <>
                  <ReferenceArea y1={0} y2={5} fill="#00ff88" fillOpacity={0.1} />
                  <ReferenceArea y1={-5} y2={0} fill="#ff3366" fillOpacity={0.1} />
                </>
              )}

              {flutterAnalysis.vgDiagrams.map((diagram, idx) => (
                <Line
                  key={idx}
                  type="monotone"
                  data={diagram.data}
                  dataKey="damping"
                  stroke={['#00c8ff', '#00ff88', '#ff8800', '#ff00ff', '#ffff00'][idx % 5]}
                  strokeWidth={diagram.modeIndex === flutterAnalysis.criticalMode ? 3 : 2}
                  dot={false}
                  name={diagram.modeName}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {displayOptions.showParametric && flutterAnalysis.parametricStudy.length > 0 && (
        <div className="parametric-section">
          <h3>Parametric Study: Thickness Sensitivity</h3>
          <div className="parametric-grid">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={flutterAnalysis.parametricStudy}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="thickness" 
                  stroke="#808080" 
                  label={{ value: 'Thickness (mm)', position: 'bottom' }}
                />
                <YAxis 
                  stroke="#808080" 
                  label={{ value: 'Flutter Speed (km/h)', angle: -90, position: 'left' }}
                />
                <Tooltip />
                <Legend />
                <ReferenceLine y={350} stroke="#ff8800" strokeDasharray="3 3" label="V_max" />
                <Line type="monotone" dataKey="flutterSpeed" stroke="#00ff88" strokeWidth={3} name="Flutter Speed" />
              </LineChart>
            </ResponsiveContainer>

            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={flutterAnalysis.parametricStudy}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="thickness" 
                  stroke="#808080" 
                  label={{ value: 'Thickness (mm)', position: 'bottom' }}
                />
                <YAxis 
                  stroke="#808080" 
                  label={{ value: 'Safety Margin', angle: -90, position: 'left' }}
                />
                <Tooltip />
                <Legend />
                <ReferenceLine y={1.2} stroke="#ff8800" strokeDasharray="3 3" label="Min" />
                <ReferenceLine y={1.5} stroke="#00ff88" strokeDasharray="3 3" label="Target" />
                <Line type="monotone" dataKey="margin" stroke="#00c8ff" strokeWidth={3} name="Safety Margin" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="display-options-section">
        <h3>Display Options</h3>
        <div className="options-grid">
          <label className="option-toggle">
            <input
              type="checkbox"
              checked={displayOptions.showIndividualModes}
              onChange={(e) => setDisplayOptions({...displayOptions, showIndividualModes: e.target.checked})}
            />
            <span>Individual Mode Diagrams</span>
          </label>

          <label className="option-toggle">
            <input
              type="checkbox"
              checked={displayOptions.showCombined}
              onChange={(e) => setDisplayOptions({...displayOptions, showCombined: e.target.checked})}
            />
            <span>Combined Diagram</span>
          </label>

          <label className="option-toggle">
            <input
              type="checkbox"
              checked={displayOptions.showParametric}
              onChange={(e) => setDisplayOptions({...displayOptions, showParametric: e.target.checked})}
            />
            <span>Parametric Study</span>
          </label>

          <label className="option-toggle">
            <input
              type="checkbox"
              checked={displayOptions.showSafetyZones}
              onChange={(e) => setDisplayOptions({...displayOptions, showSafetyZones: e.target.checked})}
            />
            <span>Safety Zones</span>
          </label>
        </div>
      </div>

      <div className="logs-section">
        <h3>📝 Analysis Logs</h3>
        <div className="logs-container">
          {logs.map((log, idx) => (
            <div key={idx} className={`log-entry log-${log.type}`}>
              <span className="log-time">[{log.timestamp}]</span>
              <span className="log-message">{log.message}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FlutterAnalysisPanel;
