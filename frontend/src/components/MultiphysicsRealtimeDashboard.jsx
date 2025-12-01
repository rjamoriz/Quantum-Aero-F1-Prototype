import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  LineChart, Line, AreaChart, Area, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import './MultiphysicsRealtimeDashboard.css';

/**
 * Dashboard Multifísica en Tiempo Real
 * Incluye: Aeroelástica, Vibración, Térmico, Aeroacústica
 * Con visualizaciones avanzadas y logs de cálculos
 */
const MultiphysicsRealtimeDashboard = () => {
  const [activePhysics, setActivePhysics] = useState({
    aeroelastic: true,
    vibration: true,
    thermal: true,
    aeroacoustic: true,
  });

  const [realtimeData, setRealtimeData] = useState({
    aeroelastic: {
      flutterSpeed: 0,
      flutterMargin: 0,
      modalFrequencies: [],
      dampingRatios: [],
      displacement: [],
    },
    vibration: {
      acceleration: [],
      velocity: [],
      displacement: [],
      fft: [],
      resonancePeaks: [],
    },
    thermal: {
      temperatures: [],
      heatFlux: [],
      thermalStress: [],
      coolingEfficiency: 0,
    },
    aeroacoustic: {
      spl: [],
      spectrum: [],
      totalNoise: 0,
      fiaCompliant: true,
    },
  });

  const [calculations, setCalculations] = useState({
    isRunning: false,
    currentStep: 0,
    totalSteps: 100,
    logs: [],
    startTime: null,
  });

  const [config, setConfig] = useState({
    velocity: 300, // km/h
    updateInterval: 1000, // ms
    simulationTime: 10, // segundos
    component: 'front_wing',
  });

  const intervalRef = useRef(null);

  // Agregar log
  const addLog = (message, type = 'info', data = null) => {
    const timestamp = new Date().toLocaleTimeString();
    setCalculations(prev => ({
      ...prev,
      logs: [
        { timestamp, message, type, data },
        ...prev.logs
      ].slice(0, 100)
    }));
  };

  // Iniciar simulación multifísica
  const startSimulation = async () => {
    setCalculations(prev => ({
      ...prev,
      isRunning: true,
      currentStep: 0,
      startTime: Date.now(),
      logs: [],
    }));

    addLog('🚀 Iniciando simulación multifísica en tiempo real', 'success');
    
    // Logs iniciales para cada física activa
    if (activePhysics.aeroelastic) {
      addLog('〰️ Módulo Aeroelástico: Inicializando análisis modal');
      addLog(`📊 Calculando ${5} primeras frecuencias naturales`);
    }
    if (activePhysics.vibration) {
      addLog('🌊 Módulo Vibración: Configurando integración temporal');
      addLog('⏱️ Método: Newmark-β, dt = 0.001s');
    }
    if (activePhysics.thermal) {
      addLog('🔥 Módulo Térmico: Inicializando transferencia de calor');
      addLog('❄️ Condiciones de enfriamiento configuradas');
    }
    if (activePhysics.aeroacoustic) {
      addLog('🔊 Módulo Aeroacústico: Método FW-H activado');
      addLog('📏 Límite FIA: 110 dB');
    }

    // Simular pasos de tiempo
    intervalRef.current = setInterval(() => {
      setCalculations(prev => {
        const newStep = prev.currentStep + 1;

        if (newStep >= config.simulationTime * 10) {
          clearInterval(intervalRef.current);
          addLog('✅ Simulación completada exitosamente', 'success');
          return { ...prev, isRunning: false, currentStep: newStep };
        }

        // Simular cálculos multifísica
        updatePhysicsData(newStep);

        return { ...prev, currentStep: newStep };
      });
    }, config.updateInterval / 10);
  };

  // Actualizar datos de física
  const updatePhysicsData = (step) => {
    const time = step * 0.1;

    setRealtimeData(prev => {
      const newData = { ...prev };

      // Aeroelástica
      if (activePhysics.aeroelastic) {
        const flutter = 280 + Math.sin(time * 0.5) * 20;
        const margin = flutter / (config.velocity * 1.1);
        
        newData.aeroelastic = {
          flutterSpeed: flutter,
          flutterMargin: margin,
          modalFrequencies: [15.2, 22.5, 35.8, 48.3, 62.1],
          dampingRatios: [0.025, 0.018, 0.032, 0.015, 0.028],
          displacement: [
            ...prev.aeroelastic.displacement,
            { time, value: 0.01 * Math.sin(time * 2 * Math.PI * 20) }
          ].slice(-100),
        };

        if (step % 10 === 0) {
          addLog(`〰️ Flutter: Vf = ${flutter.toFixed(1)} km/h, Margen = ${margin.toFixed(2)}`, 
                 margin > 1.2 ? 'success' : 'warning');
        }
      }

      // Vibración
      if (activePhysics.vibration) {
        const accel = 5 * Math.sin(time * 2 * Math.PI * 15) + 
                     2 * Math.sin(time * 2 * Math.PI * 22) +
                     Math.random() * 0.5;

        newData.vibration = {
          acceleration: [
            ...prev.vibration.acceleration,
            { time, value: accel }
          ].slice(-100),
          velocity: [
            ...prev.vibration.velocity,
            { time, value: accel / (2 * Math.PI * 15) }
          ].slice(-100),
          displacement: [
            ...prev.vibration.displacement,
            { time, value: accel / Math.pow(2 * Math.PI * 15, 2) }
          ].slice(-100),
          resonancePeaks: [
            { frequency: 15.2, amplitude: 5.2 },
            { frequency: 22.5, amplitude: 2.8 },
            { frequency: 35.8, amplitude: 1.5 },
          ],
        };

        if (step % 20 === 0 && Math.abs(accel) > 6) {
          addLog(`⚠️ Vibración: Pico detectado ${accel.toFixed(2)} m/s²`, 'warning', { accel, time });
        }
      }

      // Térmico
      if (activePhysics.thermal) {
        const brakeTemp = 650 + Math.sin(time * 0.3) * 100;
        const floorTemp = 85 + Math.sin(time * 0.2) * 15;

        newData.thermal = {
          temperatures: [
            { component: 'Freno Delantero', temp: brakeTemp, limit: 1000 },
            { component: 'Freno Trasero', temp: brakeTemp * 0.8, limit: 1000 },
            { component: 'Piso', temp: floorTemp, limit: 150 },
            { component: 'Ala Delantera', temp: 45 + Math.random() * 10, limit: 200 },
            { component: 'Difusor', temp: 120 + Math.random() * 20, limit: 200 },
          ],
          heatFlux: [
            ...prev.thermal.heatFlux,
            { time, brake: brakeTemp * 10, floor: floorTemp * 5 }
          ].slice(-100),
          thermalStress: [
            { component: 'Piso', stress: floorTemp * 0.5, limit: 150 },
          ],
          coolingEfficiency: 75 + Math.sin(time * 0.1) * 15,
        };

        if (step % 15 === 0) {
          addLog(`🔥 Térmico: T_freno = ${brakeTemp.toFixed(1)}°C, T_piso = ${floorTemp.toFixed(1)}°C`);
        }
      }

      // Aeroacústica
      if (activePhysics.aeroacoustic) {
        const totalSPL = 95 + Math.sin(time * 0.4) * 10;
        
        newData.aeroacoustic = {
          spl: [
            ...prev.aeroacoustic.spl,
            { time, value: totalSPL }
          ].slice(-100),
          spectrum: [
            { frequency: 100, spl: 70 },
            { frequency: 500, spl: 85 },
            { frequency: 1000, spl: totalSPL },
            { frequency: 2000, spl: 82 },
            { frequency: 5000, spl: 75 },
          ],
          totalNoise: totalSPL,
          fiaCompliant: totalSPL < 110,
        };

        if (step % 25 === 0) {
          const compliant = totalSPL < 110 ? '✅' : '❌';
          addLog(`🔊 Acústica: SPL = ${totalSPL.toFixed(1)} dB ${compliant}`, 
                 totalSPL < 110 ? 'success' : 'error');
        }
      }

      return newData;
    });
  };

  // Detener simulación
  const stopSimulation = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    setCalculations(prev => ({ ...prev, isRunning: false }));
    addLog('⏸️ Simulación detenida', 'warning');
  };

  // Exportar datos
  const exportData = () => {
    const data = {
      config,
      activePhysics,
      realtimeData,
      calculations: {
        totalSteps: calculations.currentStep,
        duration: calculations.startTime ? (Date.now() - calculations.startTime) / 1000 : 0,
      },
      logs: calculations.logs,
      timestamp: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `multiphysics_${Date.now()}.json`;
    link.click();
    addLog('💾 Datos exportados', 'success');
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return (
    <div className="multiphysics-realtime-dashboard">
      <div className="dashboard-header">
        <h2>⚛️ Dashboard Multifísica en Tiempo Real</h2>
        <p>Simulación acoplada: Aeroelástica • Vibración • Térmico • Aeroacústica</p>
      </div>

      {/* Controles */}
      <div className="controls-section">
        <div className="physics-toggles">
          <label className={activePhysics.aeroelastic ? 'active' : ''}>
            <input
              type="checkbox"
              checked={activePhysics.aeroelastic}
              onChange={(e) => setActivePhysics({...activePhysics, aeroelastic: e.target.checked})}
              disabled={calculations.isRunning}
            />
            〰️ Aeroelástica
          </label>
          <label className={activePhysics.vibration ? 'active' : ''}>
            <input
              type="checkbox"
              checked={activePhysics.vibration}
              onChange={(e) => setActivePhysics({...activePhysics, vibration: e.target.checked})}
              disabled={calculations.isRunning}
            />
            🌊 Vibración
          </label>
          <label className={activePhysics.thermal ? 'active' : ''}>
            <input
              type="checkbox"
              checked={activePhysics.thermal}
              onChange={(e) => setActivePhysics({...activePhysics, thermal: e.target.checked})}
              disabled={calculations.isRunning}
            />
            🔥 Térmico
          </label>
          <label className={activePhysics.aeroacoustic ? 'active' : ''}>
            <input
              type="checkbox"
              checked={activePhysics.aeroacoustic}
              onChange={(e) => setActivePhysics({...activePhysics, aeroacoustic: e.target.checked})}
              disabled={calculations.isRunning}
            />
            🔊 Aeroacústica
          </label>
        </div>

        <div className="simulation-controls">
          <div className="config-input">
            <label>Velocidad (km/h):</label>
            <input
              type="number"
              value={config.velocity}
              onChange={(e) => setConfig({...config, velocity: parseInt(e.target.value)})}
              disabled={calculations.isRunning}
            />
          </div>

          <div className="action-buttons">
            {!calculations.isRunning ? (
              <button className="btn-start" onClick={startSimulation}>
                ▶️ Iniciar Simulación
              </button>
            ) : (
              <button className="btn-stop" onClick={stopSimulation}>
                ⏸️ Detener
              </button>
            )}
            <button className="btn-export" onClick={exportData}>
              💾 Exportar
            </button>
          </div>
        </div>

        {calculations.isRunning && (
          <div className="progress-indicator">
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{width: `${(calculations.currentStep / (config.simulationTime * 10)) * 100}%`}}
              />
            </div>
            <span>Paso {calculations.currentStep} / {config.simulationTime * 10}</span>
            <span>Tiempo: {(calculations.currentStep * 0.1).toFixed(1)}s</span>
          </div>
        )}
      </div>

      {/* Visualizaciones */}
      <div className="visualizations-grid">
        {/* Aeroelástica */}
        {activePhysics.aeroelastic && (
          <div className="viz-panel aeroelastic">
            <h3>〰️ Análisis Aeroelástico</h3>
            <div className="metrics-row">
              <div className="metric-card">
                <div className="metric-label">Vel. Flutter</div>
                <div className="metric-value">{realtimeData.aeroelastic.flutterSpeed.toFixed(1)}</div>
                <div className="metric-unit">km/h</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Margen</div>
                <div className={`metric-value ${realtimeData.aeroelastic.flutterMargin > 1.2 ? 'safe' : 'warning'}`}>
                  {realtimeData.aeroelastic.flutterMargin.toFixed(2)}
                </div>
                <div className="metric-unit">x</div>
              </div>
            </div>

            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={realtimeData.aeroelastic.displacement}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,200,255,0.1)" />
                <XAxis dataKey="time" stroke="#808080" />
                <YAxis stroke="#808080" />
                <Tooltip contentStyle={{background: 'rgba(0,0,0,0.8)', border: '1px solid #00c8ff'}} />
                <Line type="monotone" dataKey="value" stroke="#00c8ff" strokeWidth={2} dot={false} name="Desplaz. (m)" />
              </LineChart>
            </ResponsiveContainer>

            <div className="modal-frequencies">
              <h4>Frecuencias Naturales</h4>
              {realtimeData.aeroelastic.modalFrequencies.map((freq, idx) => (
                <div key={idx} className="freq-item">
                  <span>Modo {idx + 1}:</span>
                  <span>{freq.toFixed(1)} Hz</span>
                  <span>ζ = {(realtimeData.aeroelastic.dampingRatios[idx] * 100).toFixed(2)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Vibración */}
        {activePhysics.vibration && (
          <div className="viz-panel vibration">
            <h3>🌊 Análisis de Vibración</h3>
            
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={realtimeData.vibration.acceleration}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,255,136,0.1)" />
                <XAxis dataKey="time" stroke="#808080" />
                <YAxis stroke="#808080" />
                <Tooltip contentStyle={{background: 'rgba(0,0,0,0.8)', border: '1px solid #00ff88'}} />
                <Line type="monotone" dataKey="value" stroke="#00ff88" strokeWidth={2} dot={false} name="Aceleración (m/s²)" />
              </LineChart>
            </ResponsiveContainer>

            <div className="resonance-peaks">
              <h4>Picos de Resonancia</h4>
              {realtimeData.vibration.resonancePeaks.map((peak, idx) => (
                <div key={idx} className="peak-item">
                  <span>{peak.frequency.toFixed(1)} Hz</span>
                  <div className="peak-bar">
                    <div className="peak-fill" style={{width: `${(peak.amplitude / 6) * 100}%`}} />
                  </div>
                  <span>{peak.amplitude.toFixed(2)} m/s²</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Térmico */}
        {activePhysics.thermal && (
          <div className="viz-panel thermal">
            <h3>🔥 Análisis Térmico</h3>
            
            <div className="temperature-bars">
              {realtimeData.thermal.temperatures.map((item, idx) => (
                <div key={idx} className="temp-item">
                  <div className="temp-label">{item.component}</div>
                  <div className="temp-bar">
                    <div 
                      className={`temp-fill ${item.temp > item.limit * 0.9 ? 'critical' : ''}`}
                      style={{width: `${(item.temp / item.limit) * 100}%`}}
                    />
                  </div>
                  <div className="temp-value">{item.temp.toFixed(0)}°C</div>
                </div>
              ))}
            </div>

            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={realtimeData.thermal.heatFlux}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,136,0,0.1)" />
                <XAxis dataKey="time" stroke="#808080" />
                <YAxis stroke="#808080" />
                <Tooltip contentStyle={{background: 'rgba(0,0,0,0.8)', border: '1px solid #ff8800'}} />
                <Area type="monotone" dataKey="brake" stackId="1" stroke="#ff8800" fill="#ff8800" fillOpacity={0.6} name="Freno (W/m²)" />
                <Area type="monotone" dataKey="floor" stackId="1" stroke="#ffaa00" fill="#ffaa00" fillOpacity={0.4} name="Piso (W/m²)" />
              </AreaChart>
            </ResponsiveContainer>

            <div className="cooling-efficiency">
              <span>Eficiencia Enfriamiento:</span>
              <span className="efficiency-value">{realtimeData.thermal.coolingEfficiency.toFixed(1)}%</span>
            </div>
          </div>
        )}

        {/* Aeroacústica */}
        {activePhysics.aeroacoustic && (
          <div className="viz-panel aeroacoustic">
            <h3>🔊 Análisis Aeroacústico</h3>
            
            <div className="noise-level">
              <div className="noise-gauge">
                <div className="gauge-label">SPL Total</div>
                <div className="gauge-value">{realtimeData.aeroacoustic.totalNoise.toFixed(1)}</div>
                <div className="gauge-unit">dB</div>
                <div className={`fia-status ${realtimeData.aeroacoustic.fiaCompliant ? 'compliant' : 'non-compliant'}`}>
                  {realtimeData.aeroacoustic.fiaCompliant ? '✅ FIA Compliant' : '❌ Excede Límite'}
                </div>
              </div>
            </div>

            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={realtimeData.aeroacoustic.spl}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,0,255,0.1)" />
                <XAxis dataKey="time" stroke="#808080" />
                <YAxis stroke="#808080" domain={[80, 120]} />
                <Tooltip contentStyle={{background: 'rgba(0,0,0,0.8)', border: '1px solid #ff00ff'}} />
                <Line type="monotone" dataKey="value" stroke="#ff00ff" strokeWidth={2} dot={false} name="SPL (dB)" />
                <Line type="monotone" dataKey={() => 110} stroke="#ff0000" strokeWidth={1} strokeDasharray="5 5" name="Límite FIA" />
              </LineChart>
            </ResponsiveContainer>

            <div className="spectrum-chart">
              <h4>Espectro de Frecuencia</h4>
              <ResponsiveContainer width="100%" height={150}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,0,255,0.1)" />
                  <XAxis dataKey="frequency" stroke="#808080" name="Freq (Hz)" />
                  <YAxis dataKey="spl" stroke="#808080" name="SPL (dB)" />
                  <Tooltip contentStyle={{background: 'rgba(0,0,0,0.8)', border: '1px solid #ff00ff'}} />
                  <Scatter data={realtimeData.aeroacoustic.spectrum} fill="#ff00ff" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>

      {/* Panel de Logs */}
      <div className="logs-panel">
        <h3>📝 Registro de Cálculos en Tiempo Real</h3>
        <div className="logs-container">
          {calculations.logs.map((log, idx) => (
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

export default MultiphysicsRealtimeDashboard;
