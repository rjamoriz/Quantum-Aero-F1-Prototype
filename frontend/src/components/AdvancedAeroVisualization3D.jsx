import React, { useRef, useEffect, useState, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid, Text } from '@react-three/drei';
import * as THREE from 'three';
import './AdvancedAeroVisualization3D.css';

/**
 * Visualización 3D Avanzada de Aerodinámica
 * - Distribución de presión con mapa de colores
 * - Líneas de corriente (streamlines)
 * - Vectores de fuerza
 * - Vórtices
 * - Malla interactiva
 */

// Componente de Wing con presión
const PressureWing = ({ geometry, pressureData, colormap = 'jet', showMesh = true }) => {
  const meshRef = useRef();
  const [vertices, setVertices] = useState([]);
  const [faces, setFaces] = useState([]);
  const [colors, setColors] = useState([]);

  useEffect(() => {
    if (!geometry || !pressureData) return;

    // Generar geometría de ala
    const { span, chord } = geometry;
    const nx = 40, ny = 20; // Resolución de malla
    const verts = [];
    const cols = [];

    for (let i = 0; i <= ny; i++) {
      for (let j = 0; j <= nx; j++) {
        const x = (j / nx) * chord;
        const y = ((i / ny) - 0.5) * span;
        
        // Perfil NACA simplificado
        const t = 0.12; // Espesor 12%
        const xn = x / chord;
        const z = t * chord * (0.2969 * Math.sqrt(xn) - 0.1260 * xn - 
                   0.3516 * xn * xn + 0.2843 * xn * xn * xn - 0.1015 * xn * xn * xn * xn);
        
        verts.push(x, y, z);

        // Color basado en presión
        const idx = i * (nx + 1) + j;
        const pressure = pressureData[idx] || 0;
        const color = getColorFromPressure(pressure, colormap);
        cols.push(color.r, color.g, color.b);
      }
    }

    setVertices(new Float32Array(verts));
    setColors(new Float32Array(cols));

  }, [geometry, pressureData, colormap]);

  // Convertir presión a color
  const getColorFromPressure = (pressure, scheme) => {
    // Normalizar presión de -3 a 1
    const normalized = (pressure + 3) / 4;
    const clamped = Math.max(0, Math.min(1, normalized));

    if (scheme === 'jet') {
      // Jet colormap: azul -> cyan -> verde -> amarillo -> rojo
      if (clamped < 0.25) {
        return { r: 0, g: clamped * 4, b: 1 };
      } else if (clamped < 0.5) {
        return { r: 0, g: 1, b: 1 - (clamped - 0.25) * 4 };
      } else if (clamped < 0.75) {
        return { r: (clamped - 0.5) * 4, g: 1, b: 0 };
      } else {
        return { r: 1, g: 1 - (clamped - 0.75) * 4, b: 0 };
      }
    }

    return { r: clamped, g: 0, b: 1 - clamped };
  };

  if (vertices.length === 0) return null;

  return (
    <mesh ref={meshRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={vertices.length / 3}
          array={vertices}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={colors.length / 3}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        wireframe={showMesh}
        metalness={0.3}
        roughness={0.7}
      />
    </mesh>
  );
};

// Componente de Streamlines
const Streamlines = ({ flowField, numLines = 20, color = '#00ffff' }) => {
  const lines = useMemo(() => {
    if (!flowField) return [];

    const streamlines = [];
    const { velocityField, bounds } = flowField;

    // Generar puntos semilla
    for (let i = 0; i < numLines; i++) {
      const y = (Math.random() - 0.5) * 2;
      const z = Math.random() * 0.5;
      const points = integrateStreamline([0, y, z], velocityField, bounds, 50);
      streamlines.push(points);
    }

    return streamlines;
  }, [flowField, numLines]);

  // Integrar línea de corriente (método Euler)
  const integrateStreamline = (start, velocityField, bounds, steps) => {
    const points = [new THREE.Vector3(...start)];
    let current = [...start];
    const dt = 0.05;

    for (let i = 0; i < steps; i++) {
      const velocity = getVelocity(current, velocityField);
      current = [
        current[0] + velocity[0] * dt,
        current[1] + velocity[1] * dt,
        current[2] + velocity[2] * dt,
      ];

      // Verificar límites
      if (current[0] > bounds.xMax || current[0] < bounds.xMin) break;
      
      points.push(new THREE.Vector3(...current));
    }

    return points;
  };

  const getVelocity = (pos, field) => {
    // Interpolación simple de campo de velocidad
    const baseVel = [1.0, 0, 0]; // Flujo uniforme base
    const perturbation = [
      0.1 * Math.sin(pos[1] * Math.PI),
      0.05 * Math.cos(pos[0] * Math.PI),
      0.02 * Math.sin(pos[2] * Math.PI * 2)
    ];
    return [
      baseVel[0] + perturbation[0],
      baseVel[1] + perturbation[1],
      baseVel[2] + perturbation[2]
    ];
  };

  return (
    <group>
      {lines.map((points, idx) => (
        <line key={idx}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={points.length}
              array={new Float32Array(points.flatMap(p => [p.x, p.y, p.z]))}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color={color} linewidth={2} />
        </line>
      ))}
    </group>
  );
};

// Vectores de Fuerza
const ForceVectors = ({ forces, scale = 0.1 }) => {
  if (!forces) return null;

  const Arrow = ({ start, direction, color, label }) => (
    <group>
      <arrowHelper
        args={[
          new THREE.Vector3(...direction).normalize(),
          new THREE.Vector3(...start),
          direction.length * scale,
          color,
          0.1,
          0.05
        ]}
      />
      <Text
        position={[start[0], start[1], start[2] + 0.2]}
        fontSize={0.1}
        color={color}
        anchorX="center"
        anchorY="middle"
      >
        {label}
      </Text>
    </group>
  );

  return (
    <group>
      {/* Downforce */}
      <Arrow
        start={[0.5, 0, 0.2]}
        direction={[0, 0, -forces.downforce]}
        color="#00ff88"
        label={`L: ${forces.downforce.toFixed(1)}N`}
      />
      
      {/* Drag */}
      <Arrow
        start={[0.5, 0, 0.2]}
        direction={[-forces.drag, 0, 0]}
        color="#ff8800"
        label={`D: ${forces.drag.toFixed(1)}N`}
      />

      {/* Side Force (si existe) */}
      {forces.sideForce && Math.abs(forces.sideForce) > 0.1 && (
        <Arrow
          start={[0.5, 0, 0.2]}
          direction={[0, forces.sideForce, 0]}
          color="#ff00ff"
          label={`SF: ${forces.sideForce.toFixed(1)}N`}
        />
      )}
    </group>
  );
};

// Indicadores de Vórtices
const VortexIndicators = ({ vortices }) => {
  if (!vortices || vortices.length === 0) return null;

  return (
    <group>
      {vortices.map((vortex, idx) => (
        <mesh key={idx} position={vortex.position}>
          <torusGeometry args={[0.05, 0.02, 16, 32]} />
          <meshStandardMaterial
            color="#ff00ff"
            emissive="#ff00ff"
            emissiveIntensity={0.5}
            transparent
            opacity={0.6}
          />
        </mesh>
      ))}
    </group>
  );
};

// Plano de Suelo
const GroundPlane = ({ size = 5, height = -0.1 }) => (
  <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, height]} receiveShadow>
    <planeGeometry args={[size, size]} />
    <meshStandardMaterial
      color="#1a1a2e"
      transparent
      opacity={0.3}
      side={THREE.DoubleSide}
    />
  </mesh>
);

// Barra de Color (leyenda)
const ColorLegend = ({ position, colormap = 'jet', min = -3, max = 1 }) => {
  const canvasRef = useRef();

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = 20;
    const height = 200;

    canvas.width = width;
    canvas.height = height;

    // Dibujar gradiente
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    
    if (colormap === 'jet') {
      gradient.addColorStop(0, '#ff0000');
      gradient.addColorStop(0.25, '#ffff00');
      gradient.addColorStop(0.5, '#00ff00');
      gradient.addColorStop(0.75, '#00ffff');
      gradient.addColorStop(1, '#0000ff');
    }

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    // Labels
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px Arial';
    ctx.fillText(max.toFixed(1), 25, 15);
    ctx.fillText(((max + min) / 2).toFixed(1), 25, height / 2);
    ctx.fillText(min.toFixed(1), 25, height - 5);

  }, [colormap, min, max]);

  return (
    <div className="color-legend" style={{ position: 'absolute', ...position }}>
      <canvas ref={canvasRef} />
      <div className="legend-label">Cp</div>
    </div>
  );
};

// Componente Principal
const AdvancedAeroVisualization3D = ({ 
  aeroData, 
  showPressure = true,
  showStreamlines = true,
  showForces = true,
  showVortices = true,
  showMesh = false,
  colormap = 'jet',
}) => {
  const [cameraPosition, setCameraPosition] = useState([3, 2, 2]);
  const [autoRotate, setAutoRotate] = useState(false);

  const geometry = aeroData?.geometry || { span: 1.8, chord: 0.25 };
  const pressureData = aeroData?.pressure || Array(800).fill(0).map(() => Math.random() * 4 - 3);
  const forces = aeroData?.forces || { downforce: 500, drag: 50, sideForce: 0 };
  const flowField = aeroData?.flowField || { 
    velocityField: [], 
    bounds: { xMin: -1, xMax: 2, yMin: -1, yMax: 1 } 
  };
  const vortices = aeroData?.vortices || [
    { position: [0.8, -0.8, 0.1] },
    { position: [0.8, 0.8, 0.1] }
  ];

  return (
    <div className="advanced-aero-visualization-3d">
      <div className="viz-controls-bar">
        <button 
          className={showPressure ? 'active' : ''}
          onClick={() => {/* toggle showPressure */}}
        >
          🌡️ Presión
        </button>
        <button 
          className={showStreamlines ? 'active' : ''}
          onClick={() => {/* toggle showStreamlines */}}
        >
          〰️ Streamlines
        </button>
        <button 
          className={showForces ? 'active' : ''}
          onClick={() => {/* toggle showForces */}}
        >
          ➡️ Fuerzas
        </button>
        <button 
          className={showVortices ? 'active' : ''}
          onClick={() => {/* toggle showVortices */}}
        >
          🌪️ Vórtices
        </button>
        <button 
          className={showMesh ? 'active' : ''}
          onClick={() => {/* toggle showMesh */}}
        >
          🔲 Malla
        </button>
        <button 
          className={autoRotate ? 'active' : ''}
          onClick={() => setAutoRotate(!autoRotate)}
        >
          🔄 Auto-Rotar
        </button>
      </div>

      <div className="canvas-container">
        <Canvas shadows camera={{ position: cameraPosition, fov: 50 }}>
          {/* Iluminación */}
          <ambientLight intensity={0.4} />
          <directionalLight
            position={[5, 5, 5]}
            intensity={1}
            castShadow
            shadow-mapSize-width={2048}
            shadow-mapSize-height={2048}
          />
          <pointLight position={[-5, 5, 5]} intensity={0.5} />

          {/* Controles */}
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            autoRotate={autoRotate}
            autoRotateSpeed={2}
          />

          {/* Grid de referencia */}
          <Grid
            args={[10, 10]}
            cellSize={0.5}
            cellThickness={0.5}
            cellColor="#6e6e6e"
            sectionSize={2}
            sectionThickness={1}
            sectionColor="#9d9d9d"
            fadeDistance={25}
            fadeStrength={1}
            followCamera={false}
            position={[0, 0, -0.1]}
          />

          {/* Plano de suelo */}
          <GroundPlane />

          {/* Wing con presión */}
          {showPressure && (
            <PressureWing
              geometry={geometry}
              pressureData={pressureData}
              colormap={colormap}
              showMesh={showMesh}
            />
          )}

          {/* Líneas de corriente */}
          {showStreamlines && (
            <Streamlines flowField={flowField} numLines={30} color="#00ffff" />
          )}

          {/* Vectores de fuerza */}
          {showForces && <ForceVectors forces={forces} scale={0.002} />}

          {/* Vórtices */}
          {showVortices && <VortexIndicators vortices={vortices} />}

          {/* Ejes de coordenadas */}
          <axesHelper args={[1]} />
        </Canvas>

        {/* Leyenda de colores */}
        {showPressure && (
          <ColorLegend
            position={{ right: '20px', top: '20px' }}
            colormap={colormap}
            min={-3}
            max={1}
          />
        )}
      </div>

      {/* Panel de información */}
      <div className="info-panel">
        <div className="info-section">
          <h4>Geometría</h4>
          <div className="info-row">
            <span>Envergadura:</span>
            <span>{geometry.span.toFixed(2)} m</span>
          </div>
          <div className="info-row">
            <span>Cuerda:</span>
            <span>{geometry.chord.toFixed(3)} m</span>
          </div>
        </div>

        <div className="info-section">
          <h4>Fuerzas</h4>
          <div className="info-row">
            <span>Downforce:</span>
            <span className="value-highlight">{forces.downforce.toFixed(1)} N</span>
          </div>
          <div className="info-row">
            <span>Drag:</span>
            <span className="value-highlight">{forces.drag.toFixed(1)} N</span>
          </div>
          <div className="info-row">
            <span>L/D:</span>
            <span className="value-highlight">{(forces.downforce / forces.drag).toFixed(2)}</span>
          </div>
        </div>

        <div className="info-section">
          <h4>Controles</h4>
          <div className="control-hint">🖱️ Click + Arrastrar: Rotar</div>
          <div className="control-hint">🖱️ Rueda: Zoom</div>
          <div className="control-hint">⌨️ Shift + Arrastrar: Pan</div>
        </div>
      </div>
    </div>
  );
};

export default AdvancedAeroVisualization3D;
