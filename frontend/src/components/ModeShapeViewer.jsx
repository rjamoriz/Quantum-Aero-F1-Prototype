/**
 * Mode Shape Viewer Component - Advanced 3D Visualization
 * High-fidelity visualization of structural mode shapes with real-time deformation
 * Includes: Multi-color schemes, export capabilities, view presets
 */
import React, { useState, useEffect, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid } from '@react-three/drei';
import * as THREE from 'three';
import axios from 'axios';
import './ModeShapeViewer.css';

/**
 * Animated Wing Mesh with Mode Shape Deformation
 */
const AnimatedWing = ({ 
  modeShape, 
  amplitude, 
  frequency, 
  isAnimating, 
  colorScheme,
  component,
  displayMode 
}) => {
  const meshRef = useRef();
  const wireframeRef = useRef();
  const [time, setTime] = useState(0);
  const originalPositions = useRef(null);

  // Component geometries
  const geometries = {
    front_wing: { width: 1.8, depth: 0.5, height: 0.05, segments: [60, 20] },
    rear_wing: { width: 1.5, depth: 0.6, height: 0.06, segments: [50, 24] },
    floor: { width: 3.5, depth: 1.2, height: 0.02, segments: [100, 40] },
    diffuser: { width: 1.0, depth: 1.5, height: 0.08, segments: [40, 50] },
  };

  const geom = geometries[component] || geometries.front_wing;

  useFrame((state, delta) => {
    if (isAnimating && meshRef.current) {
      setTime(t => t + delta);
      
      const geometry = meshRef.current.geometry;
      const positions = geometry.attributes.position;
      const colors = geometry.attributes.color;

      if (!originalPositions.current) {
        originalPositions.current = positions.array.slice();
      }

      // Sinusoidal oscillation
      const omega = 2 * Math.PI * frequency;
      const phase = Math.sin(omega * time);

      // Update positions and colors
      for (let i = 0; i < positions.count; i++) {
        const idx = i * 3;
        const x = originalPositions.current[idx];
        const y = originalPositions.current[idx + 1];
        const z = originalPositions.current[idx + 2];

        // Normalized position along span (x-direction)
        const normalizedX = (x + geom.width / 2) / geom.width;

        // Get mode shape amplitude at this position
        let modeAmplitude = 0;
        if (modeShape && modeShape.length > 0) {
          // Interpolate mode shape
          const index = Math.floor(normalizedX * (modeShape.length - 1));
          const nextIndex = Math.min(index + 1, modeShape.length - 1);
          const t = (normalizedX * (modeShape.length - 1)) - index;
          modeAmplitude = modeShape[index] * (1 - t) + modeShape[nextIndex] * t;
        } else {
          // Default: sinusoidal mode shape
          const lambda = [1.875, 4.694, 7.855, 10.996, 14.137][0];
          modeAmplitude = Math.sin(lambda * normalizedX);
        }

        // Apply deformation
        const deformation = modeAmplitude * phase * amplitude;

        positions.setXYZ(i, x, y, z + deformation);

        // Update colors based on scheme
        const color = getColorForValue(modeAmplitude * phase, colorScheme);
        colors.setXYZ(i, color.r, color.g, color.b);
      }

      positions.needsUpdate = true;
      colors.needsUpdate = true;
      geometry.computeVertexNormals();
    }
  });

  // Color mapping function
  const getColorForValue = (value, scheme) => {
    const normalized = (value + 1) / 2; // Map [-1,1] to [0,1]

    if (scheme === 'displacement') {
      // Blue -> Green -> Red
      if (normalized < 0.5) {
        return new THREE.Color(0, normalized * 2, 1 - normalized * 2);
      } else {
        return new THREE.Color((normalized - 0.5) * 2, 1 - (normalized - 0.5) * 2, 0);
      }
    } else if (scheme === 'stress') {
      // Green -> Yellow -> Red
      const absVal = Math.abs(value);
      if (absVal < 0.5) {
        return new THREE.Color(absVal * 2, 1, 0);
      } else {
        return new THREE.Color(1, 1 - (absVal - 0.5) * 2, 0);
      }
    } else if (scheme === 'velocity') {
      // Cyan -> Magenta
      const absVal = Math.abs(value);
      return new THREE.Color(absVal, 1 - absVal, 1);
    }

    return new THREE.Color(0.5, 0.5, 0.5);
  };

  // Create geometry with vertex colors
  const createGeometry = () => {
    const geometry = new THREE.PlaneGeometry(
      geom.width,
      geom.depth,
      geom.segments[0],
      geom.segments[1]
    );

    const colors = new Float32Array(geometry.attributes.position.count * 3);
    for (let i = 0; i < colors.length; i += 3) {
      colors[i] = 0.5;     // R
      colors[i + 1] = 0.5; // G
      colors[i + 2] = 0.5; // B
    }

    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    return geometry;
  };

  return (
    <>
      {/* Main deformed mesh */}
      <mesh 
        ref={meshRef} 
        rotation={[-Math.PI / 2, 0, 0]}
        castShadow
        receiveShadow
      >
        <primitive object={createGeometry()} attach="geometry" />
        <meshStandardMaterial 
          vertexColors 
          side={THREE.DoubleSide}
          metalness={0.3}
          roughness={0.4}
          wireframe={displayMode === 'wireframe'}
        />
      </mesh>

      {/* Undeformed wireframe overlay */}
      {displayMode === 'both' && (
        <mesh ref={wireframeRef} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[geom.width, geom.depth, geom.segments[0], geom.segments[1]]} />
          <meshBasicMaterial 
            color="#666666" 
            wireframe 
            transparent 
            opacity={0.2} 
          />
        </mesh>
      )}
    </>
  );
};

/**
 * Main Mode Shape Viewer Component
 */
const ModeShapeViewer = ({ modalData = null, config = {} }) => {
  const [modes, setModes] = useState([]);
  const [selectedMode, setSelectedMode] = useState(0);
  const [amplitude, setAmplitude] = useState(0.15);
  const [animationSpeed, setAnimationSpeed] = useState(1.0);
  const [isAnimating, setIsAnimating] = useState(true);
  const [modeData, setModeData] = useState(null);
  const [colorScheme, setColorScheme] = useState('displacement');
  const [displayMode, setDisplayMode] = useState('deformed');
  const [viewPreset, setViewPreset] = useState('iso');
  const [component, setComponent] = useState(config.component || 'front_wing');

  const canvasRef = useRef(null);

  useEffect(() => {
    if (modalData) {
      // Use provided modal data
      const modesFromData = modalData.frequencies.map((freq, idx) => ({
        id: idx,
        type: idx === 0 ? 'bending' : idx === 1 ? 'torsion' : idx === 2 ? 'bending' : 'coupled',
        frequency: freq,
        damping: modalData.dampingRatios[idx],
        description: `Mode ${idx + 1}`,
        modeShape: modalData.modeShapes[idx]?.map(p => p.amplitude) || [],
      }));
      setModes(modesFromData);
      setModeData(modesFromData[0]);
    } else {
      loadModes();
    }
  }, [modalData]);

  const loadModes = async () => {
    try {
      const response = await axios.get('http://localhost:3001/api/aeroelastic/modes');
      setModes(response.data.modes);
      if (response.data.modes.length > 0) {
        setModeData(response.data.modes[0]);
      }
    } catch (error) {
      // Use default modes if API not available
      const defaultModes = [
        { 
          id: 0, type: 'bending', frequency: 25.3, damping: 0.02, 
          description: '1st Bending Mode',
          modeShape: Array(21).fill(0).map((_, i) => Math.sin(1.875 * i / 20))
        },
        { 
          id: 1, type: 'torsion', frequency: 42.7, damping: 0.025, 
          description: '1st Torsion Mode',
          modeShape: Array(21).fill(0).map((_, i) => Math.sin(4.694 * i / 20))
        },
        { 
          id: 2, type: 'bending', frequency: 58.1, damping: 0.022, 
          description: '2nd Bending Mode',
          modeShape: Array(21).fill(0).map((_, i) => Math.sin(7.855 * i / 20))
        },
        { 
          id: 3, type: 'coupled', frequency: 73.5, damping: 0.028, 
          description: 'Bending-Torsion Coupled',
          modeShape: Array(21).fill(0).map((_, i) => Math.sin(10.996 * i / 20))
        },
        { 
          id: 4, type: 'local', frequency: 95.2, damping: 0.03, 
          description: 'Endplate Local Mode',
          modeShape: Array(21).fill(0).map((_, i) => Math.sin(14.137 * i / 20))
        }
      ];
      setModes(defaultModes);
      setModeData(defaultModes[0]);
    }
  };

  const handleModeChange = (modeIndex) => {
    setSelectedMode(modeIndex);
    setModeData(modes[modeIndex]);
  };

  const exportModeData = () => {
    const data = {
      mode: selectedMode + 1,
      frequency: modeData.frequency,
      damping: modeData.damping,
      modeShape: modeData.modeShape,
      timestamp: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `mode_${selectedMode + 1}_${Date.now()}.json`;
    link.click();
  };

  const takeScreenshot = () => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current.querySelector('canvas');
    if (canvas) {
      canvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `mode_${selectedMode + 1}_screenshot_${Date.now()}.png`;
        link.click();
      });
    }
  };

  // Camera positions for view presets
  const cameraPositions = {
    iso: [4, 3, 4],
    front: [0, 0, 5],
    side: [5, 0, 0],
    top: [0, 5, 0],
  };

  return (
    <div className="mode-shape-viewer-advanced">
      <div className="viewer-header">
        <h2>🎨 3D Mode Shape Viewer</h2>
        {modeData && (
          <div className="mode-info-header">
            <span className="mode-badge">Mode {selectedMode + 1}</span>
            <span className="freq-badge">{modeData.frequency.toFixed(2)} Hz</span>
            <span className="damping-badge">ζ = {(modeData.damping * 100).toFixed(2)}%</span>
          </div>
        )}
      </div>

      <div className="viewer-main">
        {/* 3D Canvas */}
        <div ref={canvasRef} className="canvas-container">
          <Canvas 
            shadows
            camera={{ position: cameraPositions[viewPreset], fov: 50 }}
          >
            <color attach="background" args={['#0a0e27']} />
            <fog attach="fog" args={['#0a0e27', 10, 50]} />
            
            {/* Lighting */}
            <ambientLight intensity={0.4} />
            <directionalLight 
              position={[5, 5, 5]} 
              intensity={0.8} 
              castShadow 
              shadow-mapSize-width={2048}
              shadow-mapSize-height={2048}
            />
            <directionalLight position={[-5, 2, -5]} intensity={0.3} color="#00c8ff" />
            <pointLight position={[0, -3, 5]} intensity={0.2} color="#00ff88" />

            {/* Animated Wing */}
            {modeData && (
              <AnimatedWing
                modeShape={modeData.modeShape}
                amplitude={amplitude}
                frequency={modeData.frequency * animationSpeed}
                isAnimating={isAnimating}
                colorScheme={colorScheme}
                component={component}
                displayMode={displayMode}
              />
            )}

            {/* Grid and Controls */}
            <Grid 
              args={[10, 10]} 
              cellColor="#003322"
              sectionColor="#00ff88"
              fadeDistance={30}
              fadeStrength={1}
              position={[0, -0.5, 0]}
            />
            <OrbitControls 
              enableDamping 
              dampingFactor={0.05}
              minDistance={2}
              maxDistance={15}
            />
            <axesHelper args={[2]} />
          </Canvas>
        </div>

        {/* Control Panel */}
        <div className="control-panel">
          {/* Mode Selection */}
          <div className="control-section">
            <label>Mode Selection</label>
            <div className="mode-grid">
              {modes.map((mode, idx) => (
                <button
                  key={idx}
                  className={`mode-btn ${selectedMode === idx ? 'active' : ''}`}
                  onClick={() => handleModeChange(idx)}
                >
                  <span className="mode-num">{idx + 1}</span>
                  <span className="mode-freq">{mode.frequency.toFixed(1)}Hz</span>
                </button>
              ))}
            </div>
          </div>

          {/* Animation Controls */}
          <div className="control-section">
            <label>Animation</label>
            <button 
              className="toggle-btn"
              onClick={() => setIsAnimating(!isAnimating)}
            >
              {isAnimating ? '⏸️ Pause' : '▶️ Play'}
            </button>
            
            <div className="slider-group">
              <span>Speed: {animationSpeed.toFixed(1)}x</span>
              <input
                type="range"
                min="0.1"
                max="3"
                step="0.1"
                value={animationSpeed}
                onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
              />
            </div>

            <div className="slider-group">
              <span>Amplification: {(amplitude * 100).toFixed(0)}x</span>
              <input
                type="range"
                min="0.05"
                max="0.5"
                step="0.05"
                value={amplitude}
                onChange={(e) => setAmplitude(parseFloat(e.target.value))}
              />
            </div>
          </div>

          {/* Display Options */}
          <div className="control-section">
            <label>Display</label>
            <select value={displayMode} onChange={(e) => setDisplayMode(e.target.value)}>
              <option value="deformed">Deformed</option>
              <option value="both">Deformed + Undeformed</option>
              <option value="wireframe">Wireframe</option>
            </select>

            <select value={colorScheme} onChange={(e) => setColorScheme(e.target.value)}>
              <option value="displacement">Displacement</option>
              <option value="stress">Stress</option>
              <option value="velocity">Velocity</option>
            </select>
          </div>

          {/* View Presets */}
          <div className="control-section">
            <label>View</label>
            <div className="view-grid">
              {['iso', 'front', 'side', 'top'].map(view => (
                <button
                  key={view}
                  className={`view-btn ${viewPreset === view ? 'active' : ''}`}
                  onClick={() => setViewPreset(view)}
                >
                  {view.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Export */}
          <div className="control-section">
            <label>Export</label>
            <div className="export-grid">
              <button className="export-btn" onClick={exportModeData}>
                💾 Data
              </button>
              <button className="export-btn" onClick={takeScreenshot}>
                📸 Image
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Mode Info Footer */}
      {modeData && (
        <div className="mode-details">
          <div className="detail-item">
            <span className="detail-label">Type:</span>
            <span className="detail-value">{modeData.type}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Description:</span>
            <span className="detail-value">{modeData.description}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Natural Frequency:</span>
            <span className="detail-value">{modeData.frequency.toFixed(2)} Hz</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Damping Ratio:</span>
            <span className="detail-value">{(modeData.damping * 100).toFixed(2)}%</span>
          </div>
        </div>
      )}

      {/* Color Legend */}
      <div className="color-legend">
        <div className="legend-title">Color Scale ({colorScheme})</div>
        <div className={`legend-gradient gradient-${colorScheme}`}>
          <span>Min</span>
          <span>Max</span>
        </div>
      </div>
    </div>
  );
};

export default ModeShapeViewer;