import React, { useState } from 'react';
import AerodynamicDataGenerator from './AerodynamicDataGenerator';
import QuantumOptimizationDashboard from './QuantumOptimizationDashboard';
import AdvancedAeroVisualization3D from './AdvancedAeroVisualization3D';
import MultiphysicsRealtimeDashboard from './MultiphysicsRealtimeDashboard';
import AeroelasticAnalysisPanel from './AeroelasticAnalysisPanel';
import VibrationAnalyzer from './VibrationAnalyzer';
import ModeShapeViewer from './ModeShapeViewer';
import FlutterAnalysisPanel from './FlutterAnalysisPanel';
import ThermalAnalysisPanel from './ThermalAnalysisPanel';
import AeroacousticAnalysisPanel from './AeroacousticAnalysisPanel';
import { useAeroDataStorage } from '../utils/AeroDataStorage';
import './QuantumAeroApp.css';

/**
 * Aplicación Principal Quantum Aero F1
 * Integra todos los módulos: CFD/VLM, Quantum, Visualización 3D, Multifísica, Aeroelástica
 */
const QuantumAeroApp = () => {
  const [activeTab, setActiveTab] = useState('generator');
  const [activeAeroelasticSubTab, setActiveAeroelasticSubTab] = useState('analysis');
  const [savedDataSummary, setSavedDataSummary] = useState(null);
  const [sharedModalData, setSharedModalData] = useState(null); // For passing data between aeroelastic components
  const storage = useAeroDataStorage();

  // Actualizar resumen de datos guardados
  const updateDataSummary = async () => {
    if (!storage.isReady) return;

    try {
      const stats = await storage.getStorageStats();
      setSavedDataSummary(stats);
    } catch (error) {
      console.error('Error obteniendo estadísticas:', error);
    }
  };

  // Cargar resumen al iniciar
  React.useEffect(() => {
    updateDataSummary();
  }, [storage.isReady]);

  const tabs = [
    {
      id: 'generator',
      label: '🌊 Generador Aerodinámico',
      icon: '🌊',
      description: 'CFD & VLM',
      component: <AerodynamicDataGenerator onDataSaved={updateDataSummary} />
    },
    {
      id: 'quantum',
      label: '⚛️ Optimización Cuántica',
      icon: '⚛️',
      description: 'QAOA, VQE, Annealing',
      component: <QuantumOptimizationDashboard onOptimizationSaved={updateDataSummary} />
    },
    {
      id: 'visualization',
      label: '🎨 Visualización 3D',
      icon: '🎨',
      description: 'Presión, Streamlines, Fuerzas',
      component: <AdvancedAeroVisualization3D />
    },
    {
      id: 'aeroelastic',
      label: '🔧 Aeroelástica & Vibración',
      icon: '🔧',
      description: 'Flutter, Modal, Vibración, High-Speed Loads',
      component: renderAeroelasticSection()
    },
    {
      id: 'multiphysics',
      label: '⚡ Dashboard Multifísica',
      icon: '⚡',
      description: 'Aeroelástica, Vibración, Térmico, Aeroacústica',
      component: <MultiphysicsRealtimeDashboard />
    }
  ];

  const aeroelasticSubTabs = [
    {
      id: 'analysis',
      label: 'Análisis Aeroelástico',
      icon: '📊',
      description: 'High-Speed Aero Loads, Modal Analysis, Flutter Prediction'
    },
    {
      id: 'vibration',
      label: 'Análisis de Vibración',
      icon: '〰️',
      description: 'Time-Domain Response, FFT, Fatigue'
    },
    {
      id: 'modeshape',
      label: 'Visualización de Modos',
      icon: '🎭',
      description: '3D Mode Shapes, Real-Time Animation'
    },
    {
      id: 'flutter',
      label: 'Flutter V-g Diagrams',
      icon: '📈',
      description: 'Multi-Mode Flutter Analysis, Parametric Studies'
    },
    {
      id: 'thermal',
      label: 'Análisis Térmico',
      icon: '🌡️',
      description: 'Conjugate Heat Transfer, Thermal Stress'
    },
    {
      id: 'acoustic',
      label: 'Análisis Aeroacústico',
      icon: '🔊',
      description: 'FW-H Solver, SPL, Noise Sources'
    }
  ];

  function renderAeroelasticSection() {
    return (
      <div className="aeroelastic-section">
        {/* Sub-navigation for aeroelastic tools */}
        <div className="aeroelastic-subnav">
          {aeroelasticSubTabs.map(subTab => (
            <button
              key={subTab.id}
              className={`aeroelastic-subtab ${activeAeroelasticSubTab === subTab.id ? 'active' : ''}`}
              onClick={() => setActiveAeroelasticSubTab(subTab.id)}
            >
              <span className="subtab-icon">{subTab.icon}</span>
              <div className="subtab-content">
                <span className="subtab-label">{subTab.label}</span>
                <span className="subtab-description">{subTab.description}</span>
              </div>
            </button>
          ))}
        </div>

        {/* Component based on active sub-tab */}
        <div className="aeroelastic-content">
          {activeAeroelasticSubTab === 'analysis' && (
            <AeroelasticAnalysisPanel 
              onModalDataGenerated={setSharedModalData}
            />
          )}
          {activeAeroelasticSubTab === 'vibration' && (
            <VibrationAnalyzer 
              modalData={sharedModalData}
            />
          )}
          {activeAeroelasticSubTab === 'modeshape' && (
            <ModeShapeViewer 
              modalData={sharedModalData}
            />
          )}
          {activeAeroelasticSubTab === 'flutter' && (
            <FlutterAnalysisPanel 
              modalData={sharedModalData}
            />
          )}
          {activeAeroelasticSubTab === 'thermal' && (
            <ThermalAnalysisPanel 
              modalData={sharedModalData}
            />
          )}
          {activeAeroelasticSubTab === 'acoustic' && (
            <AeroacousticAnalysisPanel 
              modalData={sharedModalData}
            />
          )}
        </div>
      </div>
    );
  }

  const activeTabData = tabs.find(tab => tab.id === activeTab);

  return (
    <div className="quantum-aero-app">
      {/* Header Principal */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">🏎️</div>
            <div className="logo-text">
              <h1>Quantum Aero F1 Prototype</h1>
              <p className="tagline">Advanced Aerodynamic Simulation & Quantum Optimization Platform</p>
            </div>
          </div>

          {savedDataSummary && (
            <div className="data-summary">
              <div className="summary-item">
                <span className="summary-icon">📊</span>
                <div className="summary-details">
                  <span className="summary-label">VLM Resultados</span>
                  <span className="summary-value">{savedDataSummary.vlm_results || 0}</span>
                </div>
              </div>
              <div className="summary-item">
                <span className="summary-icon">🌊</span>
                <div className="summary-details">
                  <span className="summary-label">CFD Resultados</span>
                  <span className="summary-value">{savedDataSummary.cfd_results || 0}</span>
                </div>
              </div>
              <div className="summary-item">
                <span className="summary-icon">⚛️</span>
                <div className="summary-details">
                  <span className="summary-label">Optimizaciones</span>
                  <span className="summary-value">{savedDataSummary.quantum_optimizations || 0}</span>
                </div>
              </div>
              <div className="summary-item">
                <span className="summary-icon">💾</span>
                <div className="summary-details">
                  <span className="summary-label">Storage</span>
                  <span className="summary-value">
                    {savedDataSummary.storageUsed 
                      ? `${(savedDataSummary.storageUsed / 1024 / 1024).toFixed(1)} MB`
                      : 'N/A'}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="app-navigation">
        <div className="nav-tabs">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span className="tab-icon">{tab.icon}</span>
              <div className="tab-content">
                <span className="tab-label">{tab.label.split(' ').slice(1).join(' ')}</span>
                <span className="tab-description">{tab.description}</span>
              </div>
            </button>
          ))}
        </div>

        <div className="nav-actions">
          <button 
            className="refresh-btn" 
            onClick={updateDataSummary}
            title="Actualizar estadísticas"
          >
            🔄 Refresh
          </button>
        </div>
      </nav>

      {/* Content Area */}
      <main className="app-content">
        <div className="content-header">
          <div className="content-title">
            <span className="content-icon">{activeTabData.icon}</span>
            <h2>{activeTabData.label}</h2>
          </div>
          <p className="content-description">{activeTabData.description}</p>
        </div>

        <div className="content-body">
          {activeTabData.component}
        </div>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div className="footer-content">
          <div className="footer-section">
            <span className="footer-label">Quantum Computing:</span>
            <span className="footer-value">QAOA • VQE • Quantum Annealing</span>
          </div>
          <div className="footer-section">
            <span className="footer-label">Aerodynamics:</span>
            <span className="footer-value">CFD • VLM • NACA Profiles</span>
          </div>
          <div className="footer-section">
            <span className="footer-label">Multi-Physics:</span>
            <span className="footer-value">Aeroelastic • Vibration • Thermal • Aeroacoustic</span>
          </div>
          <div className="footer-section">
            <span className="footer-label">Status:</span>
            <span className="footer-status online">● Online</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default QuantumAeroApp;
