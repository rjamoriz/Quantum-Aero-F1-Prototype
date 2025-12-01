import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, BarChart, Bar, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import './VibrationAnalyzer.css';

/**
 * Vibration Analyzer Component
 * Analyzes structural vibrations induced by high-speed aerodynamic loads
 * Includes: Modal analysis, Frequency response, FFT analysis, Resonance detection
 */
const VibrationAnalyzer = () => {
  const [config, setConfig] = useState({
    component: 'front_wing',
    velocity: 300, // km/h
    duration: 10, // seconds
    samplingRate: 1000, // Hz
    excitationType: 'aerodynamic', // 'aerodynamic', 'vortex_shedding', 'road', 'combined'
  });

  const [vibrationData, setVibrationData] = useState({
    timeHistory: {
      displacement: [],
      velocity: [],
      acceleration: [],
    },
    frequencyResponse: {
      fft: [],
      psd: [], // Power Spectral Density
      resonancePeaks: [],
    },
    modalResponse: {
      modalAmplitudes: [],
      modalVelocities: [],
    },
    fatigueAnalysis: {
      stressCycles: [],
      damage: 0,
      lifeEstimate: 0,
    },
  });

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [logs, setLogs] = useState([]);

  // Modal parameters for different components
  const modalDatabase = {
    front_wing: {
      frequencies: [18.5, 32.4, 47.2, 65.8, 89.3], // Hz
      dampingRatios: [0.025, 0.020, 0.028, 0.022, 0.030],
      modalMasses: [8.5, 7.2, 6.8, 5.9, 5.1], // kg
    },
    rear_wing: {
      frequencies: [22.3, 38.7, 54.1, 72.5, 96.8],
      dampingRatios: [0.022, 0.018, 0.025, 0.020, 0.027],
      modalMasses: [12.2, 10.5, 9.8, 8.7, 7.5],
    },
    floor: {
      frequencies: [35.6, 58.2, 78.9, 102.4, 125.7],
      dampingRatios: [0.030, 0.025, 0.032, 0.028, 0.035],
      modalMasses: [18.5, 16.2, 14.8, 13.1, 11.9],
    },
    diffuser: {
      frequencies: [42.1, 68.5, 92.3, 118.7, 145.2],
      dampingRatios: [0.028, 0.022, 0.030, 0.026, 0.033],
      modalMasses: [9.8, 8.5, 7.9, 7.1, 6.4],
    },
  };

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [
      { timestamp, message, type },
      ...prev
    ].slice(0, 100));
  };

  // Generate high-speed aerodynamic excitation force
  const generateAerodynamicForce = (t, velocity) => {
    const v_ms = velocity / 3.6;
    const rho = 1.225; // kg/m³
    const q = 0.5 * rho * v_ms * v_ms; // Dynamic pressure
    
    const chord = 0.5; // m
    const span = 1.8; // m
    const area = chord * span;
    
    // Base fluctuating force from turbulence
    const turbulenceIntensity = 0.05; // 5% turbulence
    const turbulentForce = turbulenceIntensity * q * area * (Math.random() - 0.5) * 2;
    
    // Vortex shedding (if enabled)
    let vortexForce = 0;
    if (config.excitationType === 'vortex_shedding' || config.excitationType === 'combined') {
      const St = 0.2; // Strouhal number
      const D = 0.05; // Characteristic dimension (m)
      const f_shed = St * v_ms / D; // Shedding frequency
      const omega_shed = 2 * Math.PI * f_shed;
      vortexForce = 50 * Math.sin(omega_shed * t); // N
    }
    
    // Broadband buffeting (multiple frequencies)
    let buffetingForce = 0;
    const buffetingFreqs = [10, 25, 40, 55, 70]; // Hz
    buffetingFreqs.forEach(f => {
      const omega = 2 * Math.PI * f;
      const amplitude = 100 / (1 + f / 10); // Decaying with frequency
      buffetingForce += amplitude * Math.sin(omega * t + Math.random() * 2 * Math.PI);
    });
    
    // Road excitation (if enabled)
    let roadForce = 0;
    if (config.excitationType === 'road' || config.excitationType === 'combined') {
      const wheelFreq = v_ms / (2 * Math.PI * 0.33); // Wheel rotation freq
      const kerbFreq = 8; // Hz (kerbs)
      roadForce = 200 * Math.sin(2 * Math.PI * wheelFreq * t) +
                  300 * Math.sin(2 * Math.PI * kerbFreq * t);
    }
    
    return turbulentForce + vortexForce + buffetingForce + roadForce;
  };

  // Modal superposition for time integration
  const performModalSuperposition = async () => {
    addLog('🌊 Starting modal superposition analysis...', 'info');
    
    const modal = modalDatabase[config.component];
    const dt = 1 / config.samplingRate; // Time step
    const nSteps = Math.floor(config.duration / dt);
    
    // Initialize arrays
    const timeHistory = {
      displacement: [],
      velocity: [],
      acceleration: [],
    };
    
    const modalAmplitudes = modal.frequencies.map(() => 0);
    const modalVelocities = modal.frequencies.map(() => 0);
    
    // Newmark-β parameters
    const gamma = 0.5;
    const beta = 0.25;
    
    addLog(`⏱️ Time integration: ${nSteps} steps, dt = ${(dt*1000).toFixed(2)} ms`, 'info');
    
    for (let i = 0; i < nSteps; i++) {
      const t = i * dt;
      
      // Generate aerodynamic force
      const F_aero = generateAerodynamicForce(t, config.velocity);
      
      // Project force onto modal coordinates
      const modalForces = modal.frequencies.map((f_n, modeIdx) => {
        // Modal participation factor (simplified)
        const phi_participation = 1.0;
        return phi_participation * F_aero / modal.modalMasses[modeIdx];
      });
      
      // Update each mode using Newmark-β
      let totalDisplacement = 0;
      let totalVelocity = 0;
      let totalAcceleration = 0;
      
      modal.frequencies.forEach((f_n, modeIdx) => {
        const omega_n = 2 * Math.PI * f_n;
        const zeta = modal.dampingRatios[modeIdx];
        const m_n = modal.modalMasses[modeIdx];
        
        // Modal stiffness and damping
        const k_n = omega_n * omega_n * m_n;
        const c_n = 2 * zeta * omega_n * m_n;
        
        // Previous state
        const q = modalAmplitudes[modeIdx];
        const q_dot = modalVelocities[modeIdx];
        
        // Modal acceleration
        const q_ddot = (modalForces[modeIdx] - c_n * q_dot - k_n * q) / m_n;
        
        // Newmark-β update
        const q_new = q + dt * q_dot + (0.5 - beta) * dt * dt * q_ddot;
        const q_dot_new = q_dot + (1 - gamma) * dt * q_ddot;
        
        // Store updated state
        modalAmplitudes[modeIdx] = q_new;
        modalVelocities[modeIdx] = q_dot_new;
        
        // Accumulate physical response
        totalDisplacement += q_new;
        totalVelocity += q_dot_new;
        totalAcceleration += q_ddot;
      });
      
      // Store time history (sample every 10th point for performance)
      if (i % 10 === 0) {
        timeHistory.displacement.push({
          time: t,
          value: totalDisplacement * 1000, // Convert to mm
        });
        
        timeHistory.velocity.push({
          time: t,
          value: totalVelocity * 1000, // Convert to mm/s
        });
        
        timeHistory.acceleration.push({
          time: t,
          value: totalAcceleration * 9.81, // Convert to g's
        });
      }
    }
    
    addLog('✅ Modal superposition completed', 'success');
    return timeHistory;
  };

  // Perform FFT analysis
  const performFFT = (signal) => {
    addLog('📊 Computing FFT...', 'info');
    
    // Simple FFT implementation (DFT for visualization)
    const N = signal.length;
    const fft = [];
    const psd = [];
    const fs = config.samplingRate / 10; // Effective sampling rate after decimation
    
    for (let k = 0; k < N / 2; k++) {
      const freq = (k * fs) / N;
      
      if (freq > 200) break; // Limit to 200 Hz
      
      let real = 0;
      let imag = 0;
      
      for (let n = 0; n < N; n++) {
        const angle = (-2 * Math.PI * k * n) / N;
        real += signal[n].value * Math.cos(angle);
        imag += signal[n].value * Math.sin(angle);
      }
      
      const magnitude = Math.sqrt(real * real + imag * imag) / N;
      const power = magnitude * magnitude;
      
      fft.push({ frequency: freq, magnitude });
      psd.push({ frequency: freq, power });
    }
    
    // Detect resonance peaks
    const resonancePeaks = [];
    const threshold = 0.01;
    
    for (let i = 2; i < psd.length - 2; i++) {
      if (psd[i].power > threshold &&
          psd[i].power > psd[i-1].power &&
          psd[i].power > psd[i+1].power &&
          psd[i].power > psd[i-2].power &&
          psd[i].power > psd[i+2].power) {
        resonancePeaks.push({
          frequency: psd[i].frequency,
          amplitude: psd[i].power,
        });
      }
    }
    
    addLog(`🎯 Detected ${resonancePeaks.length} resonance peaks`, 'success');
    
    return { fft, psd, resonancePeaks };
  };

  // Fatigue damage calculation (Palmgren-Miner)
  const calculateFatigueDamage = (acceleration) => {
    addLog('💀 Computing fatigue damage...', 'info');
    
    // Rainflow counting algorithm (simplified)
    const stressCycles = [];
    const S_N_exponent = 10; // For carbon fiber
    const C_constant = 1e15; // Material constant
    
    // Count cycles using simple peak detection
    const peaks = [];
    for (let i = 1; i < acceleration.length - 1; i++) {
      if ((acceleration[i].value > acceleration[i-1].value && 
           acceleration[i].value > acceleration[i+1].value) ||
          (acceleration[i].value < acceleration[i-1].value && 
           acceleration[i].value < acceleration[i+1].value)) {
        peaks.push(Math.abs(acceleration[i].value));
      }
    }
    
    // Group peaks into stress ranges
    const stressRanges = {};
    for (let i = 0; i < peaks.length - 1; i++) {
      const range = Math.abs(peaks[i+1] - peaks[i]);
      const bin = Math.floor(range * 10) / 10;
      stressRanges[bin] = (stressRanges[bin] || 0) + 1;
    }
    
    // Calculate damage for each stress range
    let totalDamage = 0;
    Object.keys(stressRanges).forEach(S => {
      const n_i = stressRanges[S];
      const N_i = C_constant / Math.pow(parseFloat(S), S_N_exponent);
      const damage_i = n_i / N_i;
      totalDamage += damage_i;
      
      stressCycles.push({
        stress: parseFloat(S),
        cycles: n_i,
        allowable: N_i,
        damage: damage_i,
      });
    });
    
    // Estimate life in race cycles
    const raceTime = 2 * 3600; // 2 hours in seconds
    const cyclesPerRace = peaks.length * (raceTime / config.duration);
    const damagePerRace = totalDamage * (raceTime / config.duration);
    const racesToFailure = 1.0 / damagePerRace;
    
    addLog(`📉 Fatigue life: ${racesToFailure.toFixed(1)} races`, 
           racesToFailure > 10 ? 'success' : 'warning');
    
    return {
      stressCycles,
      damage: totalDamage,
      lifeEstimate: racesToFailure,
    };
  };

  // Main analysis workflow
  const runVibrationAnalysis = async () => {
    setIsAnalyzing(true);
    addLog('🚀 Starting vibration analysis...', 'success');
    
    try {
      // Step 1: Modal superposition
      const timeHistory = await performModalSuperposition();
      
      // Step 2: FFT analysis
      const frequencyResponse = performFFT(timeHistory.acceleration);
      
      // Step 3: Fatigue analysis
      const fatigueAnalysis = calculateFatigueDamage(timeHistory.acceleration);
      
      // Update state
      setVibrationData({
        timeHistory,
        frequencyResponse,
        modalResponse: {
          modalAmplitudes: [],
          modalVelocities: [],
        },
        fatigueAnalysis,
      });
      
      addLog('✅ Vibration analysis completed!', 'success');
      
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
      vibrationData,
      timestamp: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `vibration_analysis_${Date.now()}.json`;
    link.click();
    
    addLog('💾 Results exported', 'success');
  };

  return (
    <div className="vibration-analyzer">
      <div className="analyzer-header">
        <h2>🌊 Vibration Analyzer</h2>
        <p>High-Speed Aerodynamic Load Analysis • Modal Response • Fatigue Assessment</p>
      </div>

      {/* Configuration */}
      <div className="config-section">
        <h3>Configuration</h3>
        
        <div className="config-grid">
          <div className="config-group">
            <label>Component</label>
            <select value={config.component} onChange={(e) => setConfig({...config, component: e.target.value})}>
              <option value="front_wing">Front Wing</option>
              <option value="rear_wing">Rear Wing</option>
              <option value="floor">Floor</option>
              <option value="diffuser">Diffuser</option>
            </select>
          </div>

          <div className="config-group">
            <label>Velocity: {config.velocity} km/h</label>
            <input
              type="range"
              min="100"
              max="400"
              value={config.velocity}
              onChange={(e) => setConfig({...config, velocity: parseInt(e.target.value)})}
            />
          </div>

          <div className="config-group">
            <label>Duration: {config.duration} s</label>
            <input
              type="range"
              min="5"
              max="30"
              value={config.duration}
              onChange={(e) => setConfig({...config, duration: parseInt(e.target.value)})}
            />
          </div>

          <div className="config-group">
            <label>Excitation Type</label>
            <select value={config.excitationType} onChange={(e) => setConfig({...config, excitationType: e.target.value})}>
              <option value="aerodynamic">Aerodynamic Only</option>
              <option value="vortex_shedding">Vortex Shedding</option>
              <option value="road">Road Input</option>
              <option value="combined">Combined</option>
            </select>
          </div>
        </div>

        <div className="action-buttons">
          <button 
            className="btn-analyze" 
            onClick={runVibrationAnalysis}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? '⏳ Analyzing...' : '▶️ Run Analysis'}
          </button>
          <button className="btn-export" onClick={exportResults}>
            💾 Export
          </button>
        </div>
      </div>

      {/* Results */}
      <div className="results-grid">
        {/* Time History */}
        <div className="result-panel">
          <h3>Time Domain Response</h3>
          
          {vibrationData.timeHistory.acceleration.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={vibrationData.timeHistory.acceleration}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="time" stroke="#808080" label={{ value: 'Time (s)', position: 'bottom' }} />
                <YAxis stroke="#808080" label={{ value: 'Acceleration (g)', angle: -90, position: 'left' }} />
                <Tooltip contentStyle={{background: 'rgba(0,0,0,0.8)', border: '1px solid #00ff88'}} />
                <Line type="monotone" dataKey="value" stroke="#00ff88" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Frequency Response */}
        <div className="result-panel">
          <h3>Frequency Response (FFT)</h3>
          
          {vibrationData.frequencyResponse.psd.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={vibrationData.frequencyResponse.psd.filter((_, i) => i % 2 === 0)}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="frequency" stroke="#808080" label={{ value: 'Frequency (Hz)', position: 'bottom' }} />
                <YAxis stroke="#808080" label={{ value: 'PSD', angle: -90, position: 'left' }} />
                <Tooltip contentStyle={{background: 'rgba(0,0,0,0.8)', border: '1px solid #00c8ff'}} />
                <Bar dataKey="power" fill="#00c8ff" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Resonance Peaks */}
        <div className="result-panel">
          <h3>Resonance Peaks</h3>
          
          {vibrationData.frequencyResponse.resonancePeaks.length > 0 ? (
            <div className="peaks-list">
              {vibrationData.frequencyResponse.resonancePeaks.map((peak, idx) => (
                <div key={idx} className="peak-item">
                  <span className="peak-freq">{peak.frequency.toFixed(2)} Hz</span>
                  <div className="peak-bar">
                    <div 
                      className="peak-fill" 
                      style={{width: `${Math.min(peak.amplitude * 1000, 100)}%`}}
                    />
                  </div>
                  <span className="peak-amp">{peak.amplitude.toFixed(4)}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="no-data">No resonance peaks detected</div>
          )}
        </div>

        {/* Fatigue Analysis */}
        <div className="result-panel">
          <h3>Fatigue Analysis</h3>
          
          {vibrationData.fatigueAnalysis.damage > 0 && (
            <>
              <div className="fatigue-summary">
                <div className="fatigue-metric">
                  <span>Total Damage:</span>
                  <span className={vibrationData.fatigueAnalysis.damage > 0.1 ? 'critical' : 'safe'}>
                    {(vibrationData.fatigueAnalysis.damage * 100).toExponential(2)}%
                  </span>
                </div>
                <div className="fatigue-metric">
                  <span>Estimated Life:</span>
                  <span className={vibrationData.fatigueAnalysis.lifeEstimate > 10 ? 'safe' : 'warning'}>
                    {vibrationData.fatigueAnalysis.lifeEstimate.toFixed(1)} races
                  </span>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Logs */}
      <div className="logs-panel">
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

export default VibrationAnalyzer;
