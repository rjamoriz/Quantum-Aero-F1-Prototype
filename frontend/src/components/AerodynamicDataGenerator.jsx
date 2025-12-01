import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './AerodynamicDataGenerator.css';

/**
 * Componente para generación sintética de datos aerodinámicos
 * Soporta CFD y Vortex Lattice Method (VLM)
 * Incluye perfiles NACA de NASA y configuraciones F1
 */
const AerodynamicDataGenerator = () => {
  const [config, setConfig] = useState({
    method: 'vlm', // 'vlm' o 'cfd'
    numSamples: 100,
    airfoilSource: 'naca', // 'naca', 'nasa', 'custom'
    nacaProfile: '6412', // Perfil NACA por defecto
    velocity: 50, // m/s
    alphaRange: { min: -5, max: 25 },
    reynoldsNumber: 1e6,
    f1Component: 'front_wing', // 'front_wing', 'rear_wing', 'floor', 'diffuser'
  });

  const [generation, setGeneration] = useState({
    isRunning: false,
    progress: 0,
    currentSample: 0,
    logs: [],
    results: null,
    timeElapsed: 0,
  });

  const [nacaProfiles] = useState({
    front_wing: ['6412', '4415', '4418'],
    rear_wing: ['9618', '6412'],
    floor: ['0009'],
    diffuser: ['23012'],
  });

  const [visualization, setVisualization] = useState({
    showPressure: true,
    showStreamlines: true,
    showForces: true,
    colormap: 'jet',
  });

  // Agregar log
  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setGeneration(prev => ({
      ...prev,
      logs: [{ timestamp, message, type }, ...prev.logs].slice(0, 100)
    }));
  };

  // Generar datos con VLM
  const generateVLMData = async () => {
    try {
      setGeneration(prev => ({ ...prev, isRunning: true, progress: 0, logs: [] }));
      addLog('🚀 Iniciando generación de datos con VLM');
      addLog(`📊 Configuración: ${config.numSamples} muestras, perfil NACA ${config.nacaProfile}`);

      const startTime = Date.now();
      let interval = setInterval(() => {
        setGeneration(prev => ({ ...prev, timeElapsed: Math.floor((Date.now() - startTime) / 1000) }));
      }, 1000);

      const results = {
        samples: [],
        statistics: { cl_mean: 0, cl_std: 0, cd_mean: 0, cd_std: 0 },
      };

      // Generar muestras
      for (let i = 0; i < config.numSamples; i++) {
        const alpha = config.alphaRange.min + 
          (config.alphaRange.max - config.alphaRange.min) * Math.random();

        addLog(`🔄 Muestra ${i + 1}/${config.numSamples}: α = ${alpha.toFixed(2)}°`);

        const response = await axios.post('http://localhost:8001/vlm/solve', {
          geometry: {
            span: config.f1Component === 'front_wing' ? 1.8 : 0.75,
            chord: config.f1Component === 'front_wing' ? 0.25 : 0.35,
            twist: -2.0,
            dihedral: 0.0,
            sweep: 0.0,
            taper_ratio: 1.0,
          },
          velocity: config.velocity,
          alpha: alpha,
          yaw: 0,
          rho: 1.225,
          n_panels_x: 20,
          n_panels_y: 10,
        });

        results.samples.push({
          alpha,
          ...response.data,
          nacaProfile: config.nacaProfile,
          component: config.f1Component,
        });

        setGeneration(prev => ({
          ...prev,
          progress: ((i + 1) / config.numSamples) * 100,
          currentSample: i + 1,
        }));

        addLog(`✅ CL = ${response.data.cl.toFixed(4)}, CD = ${response.data.cd.toFixed(4)}, L/D = ${response.data.l_over_d.toFixed(2)}`);
      }

      // Calcular estadísticas
      const cls = results.samples.map(s => s.cl);
      const cds = results.samples.map(s => s.cd);
      
      results.statistics = {
        cl_mean: cls.reduce((a, b) => a + b, 0) / cls.length,
        cl_std: Math.sqrt(cls.map(x => Math.pow(x - results.statistics.cl_mean, 2)).reduce((a, b) => a + b) / cls.length),
        cd_mean: cds.reduce((a, b) => a + b, 0) / cds.length,
        cd_std: Math.sqrt(cds.map(x => Math.pow(x - results.statistics.cd_mean, 2)).reduce((a, b) => a + b) / cds.length),
      };

      clearInterval(interval);
      setGeneration(prev => ({ 
        ...prev, 
        isRunning: false, 
        progress: 100, 
        results,
        timeElapsed: Math.floor((Date.now() - startTime) / 1000)
      }));

      addLog(`🎉 Generación completada: ${config.numSamples} muestras en ${Math.floor((Date.now() - startTime) / 1000)}s`, 'success');
      addLog(`📈 CL medio: ${results.statistics.cl_mean.toFixed(4)} ± ${results.statistics.cl_std.toFixed(4)}`);
      addLog(`📈 CD medio: ${results.statistics.cd_mean.toFixed(4)} ± ${results.statistics.cd_std.toFixed(4)}`);

    } catch (error) {
      addLog(`❌ Error: ${error.message}`, 'error');
      setGeneration(prev => ({ ...prev, isRunning: false }));
    }
  };

  // Generar datos con CFD
  const generateCFDData = async () => {
    try {
      setGeneration(prev => ({ ...prev, isRunning: true, progress: 0, logs: [] }));
      addLog('🚀 Iniciando generación de datos con CFD sintético');
      
      const startTime = Date.now();
      let interval = setInterval(() => {
        setGeneration(prev => ({ ...prev, timeElapsed: Math.floor((Date.now() - startTime) / 1000) }));
      }, 1000);

      const results = {
        samples: [],
        statistics: {},
      };

      // Generar muestras CFD sintéticas
      for (let i = 0; i < config.numSamples; i++) {
        const params = {
          chord: 0.8 + Math.random() * 0.4,
          span: 1.5 + Math.random() * 1.0,
          thickness: 0.08 + Math.random() * 0.08,
          camber: 0.02 + Math.random() * 0.04,
          angle_of_attack: -5 + Math.random() * 20,
          reynolds: 5e5 + Math.random() * 1e6,
          mach: 0.2 + Math.random() * 0.2,
        };

        addLog(`🔄 CFD Muestra ${i + 1}/${config.numSamples}: α = ${params.angle_of_attack.toFixed(2)}°, Re = ${(params.reynolds / 1e6).toFixed(2)}M`);

        // Simulación sintética (valores aproximados basados en teoría)
        const cl = 0.1 * params.angle_of_attack + params.camber * 10 + (Math.random() - 0.5) * 0.1;
        const cd = 0.01 + 0.001 * Math.pow(params.angle_of_attack, 2) + (Math.random() - 0.5) * 0.004;
        const cm = -0.1 * params.camber + (Math.random() - 0.5) * 0.02;

        results.samples.push({
          ...params,
          cl,
          cd,
          cm,
          l_over_d: cl / cd,
          nacaProfile: config.nacaProfile,
          component: config.f1Component,
        });

        setGeneration(prev => ({
          ...prev,
          progress: ((i + 1) / config.numSamples) * 100,
          currentSample: i + 1,
        }));

        addLog(`✅ CL = ${cl.toFixed(4)}, CD = ${cd.toFixed(4)}, L/D = ${(cl / cd).toFixed(2)}`);
      }

      clearInterval(interval);
      setGeneration(prev => ({ 
        ...prev, 
        isRunning: false, 
        progress: 100, 
        results,
        timeElapsed: Math.floor((Date.now() - startTime) / 1000)
      }));

      addLog(`🎉 Generación CFD completada: ${config.numSamples} muestras`, 'success');

    } catch (error) {
      addLog(`❌ Error CFD: ${error.message}`, 'error');
      setGeneration(prev => ({ ...prev, isRunning: false }));
    }
  };

  // Exportar datos
  const exportData = () => {
    if (!generation.results) return;

    const dataStr = JSON.stringify(generation.results, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `aerodynamic_data_${config.method}_${Date.now()}.json`;
    link.click();
    
    addLog(`💾 Datos exportados: ${generation.results.samples.length} muestras`, 'success');
  };

  // Visualizar muestra
  const visualizeSample = (sampleIndex) => {
    if (!generation.results || !generation.results.samples[sampleIndex]) return;
    
    const sample = generation.results.samples[sampleIndex];
    addLog(`🎨 Visualizando muestra #${sampleIndex}: α = ${sample.alpha?.toFixed(2)}°`);
    // Aquí se conectaría con el componente de visualización 3D
  };

  return (
    <div className="aerodynamic-data-generator">
      <div className="generator-header">
        <h2>🌪️ Generador de Datos Aerodinámicos</h2>
        <p>Generación sintética con CFD y Vortex Lattice Method - Perfiles NACA para F1</p>
      </div>

      <div className="generator-grid">
        {/* Panel de Configuración */}
        <div className="config-panel">
          <h3>⚙️ Configuración</h3>
          
          <div className="form-group">
            <label>Método de Simulación</label>
            <select 
              value={config.method} 
              onChange={(e) => setConfig({...config, method: e.target.value})}
              disabled={generation.isRunning}
            >
              <option value="vlm">Vortex Lattice Method (VLM)</option>
              <option value="cfd">CFD Sintético</option>
            </select>
          </div>

          <div className="form-group">
            <label>Componente F1</label>
            <select 
              value={config.f1Component} 
              onChange={(e) => setConfig({
                ...config, 
                f1Component: e.target.value,
                nacaProfile: nacaProfiles[e.target.value][0]
              })}
              disabled={generation.isRunning}
            >
              <option value="front_wing">Ala Delantera</option>
              <option value="rear_wing">Ala Trasera</option>
              <option value="floor">Piso</option>
              <option value="diffuser">Difusor</option>
            </select>
          </div>

          <div className="form-group">
            <label>Perfil NACA</label>
            <select 
              value={config.nacaProfile} 
              onChange={(e) => setConfig({...config, nacaProfile: e.target.value})}
              disabled={generation.isRunning}
            >
              {nacaProfiles[config.f1Component].map(profile => (
                <option key={profile} value={profile}>NACA {profile}</option>
              ))}
            </select>
            <small>Perfiles específicos de F1</small>
          </div>

          <div className="form-group">
            <label>Número de Muestras: {config.numSamples}</label>
            <input 
              type="range" 
              min="10" 
              max="1000" 
              value={config.numSamples}
              onChange={(e) => setConfig({...config, numSamples: parseInt(e.target.value)})}
              disabled={generation.isRunning}
            />
          </div>

          <div className="form-group">
            <label>Velocidad (m/s): {config.velocity}</label>
            <input 
              type="range" 
              min="20" 
              max="100" 
              value={config.velocity}
              onChange={(e) => setConfig({...config, velocity: parseInt(e.target.value)})}
              disabled={generation.isRunning}
            />
            <small>{(config.velocity * 3.6).toFixed(0)} km/h</small>
          </div>

          <div className="form-group">
            <label>Rango α (grados)</label>
            <div className="range-inputs">
              <input 
                type="number" 
                value={config.alphaRange.min}
                onChange={(e) => setConfig({
                  ...config, 
                  alphaRange: {...config.alphaRange, min: parseFloat(e.target.value)}
                })}
                disabled={generation.isRunning}
                placeholder="Min"
              />
              <span>a</span>
              <input 
                type="number" 
                value={config.alphaRange.max}
                onChange={(e) => setConfig({
                  ...config, 
                  alphaRange: {...config.alphaRange, max: parseFloat(e.target.value)}
                })}
                disabled={generation.isRunning}
                placeholder="Max"
              />
            </div>
          </div>

          <div className="action-buttons">
            <button 
              className="btn-primary" 
              onClick={config.method === 'vlm' ? generateVLMData : generateCFDData}
              disabled={generation.isRunning}
            >
              {generation.isRunning ? '⏳ Generando...' : '🚀 Generar Datos'}
            </button>
            
            {generation.results && (
              <button className="btn-secondary" onClick={exportData}>
                💾 Exportar JSON
              </button>
            )}
          </div>

          {/* Opciones de Visualización */}
          <div className="visualization-options">
            <h4>Opciones de Visualización</h4>
            <label>
              <input 
                type="checkbox" 
                checked={visualization.showPressure}
                onChange={(e) => setVisualization({...visualization, showPressure: e.target.checked})}
              />
              Distribución de Presión
            </label>
            <label>
              <input 
                type="checkbox" 
                checked={visualization.showStreamlines}
                onChange={(e) => setVisualization({...visualization, showStreamlines: e.target.checked})}
              />
              Líneas de Corriente
            </label>
            <label>
              <input 
                type="checkbox" 
                checked={visualization.showForces}
                onChange={(e) => setVisualization({...visualization, showForces: e.target.checked})}
              />
              Vectores de Fuerza
            </label>
          </div>
        </div>

        {/* Panel de Progreso */}
        <div className="progress-panel">
          <h3>📊 Progreso</h3>
          
          {generation.isRunning && (
            <>
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{width: `${generation.progress}%`}}
                />
              </div>
              <div className="progress-stats">
                <span>{generation.currentSample} / {config.numSamples} muestras</span>
                <span>{generation.progress.toFixed(1)}%</span>
                <span>⏱️ {generation.timeElapsed}s</span>
              </div>
            </>
          )}

          {/* Logs */}
          <div className="logs-container">
            <h4>📝 Registro de Actividad</h4>
            <div className="logs-list">
              {generation.logs.map((log, idx) => (
                <div key={idx} className={`log-entry log-${log.type}`}>
                  <span className="log-time">[{log.timestamp}]</span>
                  <span className="log-message">{log.message}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Panel de Resultados */}
        {generation.results && (
          <div className="results-panel">
            <h3>📈 Resultados</h3>
            
            <div className="stats-grid">
              <div className="stat-card">
                <h4>Muestras Generadas</h4>
                <div className="stat-value">{generation.results.samples.length}</div>
              </div>
              
              {generation.results.statistics && (
                <>
                  <div className="stat-card">
                    <h4>CL Promedio</h4>
                    <div className="stat-value">
                      {generation.results.statistics.cl_mean?.toFixed(4)}
                    </div>
                    <div className="stat-detail">
                      ± {generation.results.statistics.cl_std?.toFixed(4)}
                    </div>
                  </div>
                  
                  <div className="stat-card">
                    <h4>CD Promedio</h4>
                    <div className="stat-value">
                      {generation.results.statistics.cd_mean?.toFixed(4)}
                    </div>
                    <div className="stat-detail">
                      ± {generation.results.statistics.cd_std?.toFixed(4)}
                    </div>
                  </div>
                  
                  <div className="stat-card">
                    <h4>L/D Promedio</h4>
                    <div className="stat-value">
                      {(generation.results.statistics.cl_mean / generation.results.statistics.cd_mean).toFixed(2)}
                    </div>
                  </div>
                </>
              )}
              
              <div className="stat-card">
                <h4>Tiempo Total</h4>
                <div className="stat-value">{generation.timeElapsed}s</div>
                <div className="stat-detail">
                  {(generation.timeElapsed / config.numSamples).toFixed(2)}s/muestra
                </div>
              </div>
            </div>

            {/* Lista de Muestras */}
            <div className="samples-list">
              <h4>Muestras Generadas</h4>
              <div className="samples-table-container">
                <table className="samples-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>α (°)</th>
                      <th>CL</th>
                      <th>CD</th>
                      <th>CM</th>
                      <th>L/D</th>
                      <th>Acción</th>
                    </tr>
                  </thead>
                  <tbody>
                    {generation.results.samples.slice(0, 20).map((sample, idx) => (
                      <tr key={idx}>
                        <td>{idx + 1}</td>
                        <td>{sample.alpha?.toFixed(2) || sample.angle_of_attack?.toFixed(2)}</td>
                        <td>{sample.cl?.toFixed(4)}</td>
                        <td>{sample.cd?.toFixed(4)}</td>
                        <td>{sample.cm?.toFixed(4)}</td>
                        <td>{sample.l_over_d?.toFixed(2)}</td>
                        <td>
                          <button 
                            className="btn-small" 
                            onClick={() => visualizeSample(idx)}
                          >
                            👁️ Ver
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AerodynamicDataGenerator;
