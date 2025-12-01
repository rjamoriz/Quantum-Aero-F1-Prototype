# 🎨 Guía Visual de Componentes - Quantum Aero F1

## 📋 Índice
1. [QuantumAeroApp - Aplicación Principal](#quantumaeroapp)
2. [AerodynamicDataGenerator - Generador Aerodinámico](#aerodynamicdatagenerator)
3. [QuantumOptimizationDashboard - Optimización Cuántica](#quantumoptimizationdashboard)
4. [AdvancedAeroVisualization3D - Visualización 3D](#advancedaerovisualization3d)
5. [MultiphysicsRealtimeDashboard - Dashboard Multifísica](#multiphysicsrealtimedashboard)

---

## 1. QuantumAeroApp - Aplicación Principal {#quantumaeroapp}

### Layout
```
┌─────────────────────────────────────────────────────────────┐
│ 🏎️ Quantum Aero F1 Prototype              [Stats Summary] │
│ Advanced Aerodynamic Simulation Platform                    │
├─────────────────────────────────────────────────────────────┤
│ [🌊 Generador] [⚛️ Quantum] [🎨 3D Viz] [⚡ Multiphysics]│
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                 [CONTENT AREA - Active Tab]                 │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│ Quantum: QAOA•VQE  |  Aero: CFD•VLM  |  Status: ● Online  │
└─────────────────────────────────────────────────────────────┘
```

### Características
- **Header**: Logo animado + estadísticas en vivo (VLM results, CFD results, Optimizations, Storage)
- **Navigation**: 4 tabs con iconos y descripciones
- **Content**: Área dinámica que muestra el componente activo
- **Footer**: Información del sistema y estado

### Interacciones
1. **Click en Tab**: Cambia el contenido mostrado
2. **Botón Refresh**: Actualiza estadísticas de almacenamiento
3. **Responsive**: Se adapta a mobile/tablet/desktop

---

## 2. AerodynamicDataGenerator - Generador Aerodinámico {#aerodynamicdatagenerator}

### Layout
```
┌───────────────────┬─────────────────────────┬──────────────────┐
│   CONFIGURATION   │    GENERATION STATUS    │   STATISTICS     │
│                   │                         │                  │
│ Method:           │  ▰▰▰▰▰▰▰▰▱▱ 80%        │  CL mean: 1.245  │
│ [●] VLM           │                         │  CL std:  0.123  │
│ [ ] CFD           │  Generated: 80/100      │  CD mean: 0.045  │
│                   │                         │  CD std:  0.008  │
│ Component:        │  [▶ Generate] [📊 Export] │                  │
│ [Front Wing  ▾]   │                         │                  │
│                   │  ┌─ Pressure Chart ──┐ │                  │
│ NACA Profile:     │  │                    │ │                  │
│ [NACA6412    ▾]   │  │     📈             │ │                  │
│                   │  │                    │ │                  │
│ Samples: [100  ]  │  └────────────────────┘ │                  │
│                   │                         │                  │
│ Velocity: 300 km/h│                         │                  │
│ AoA: 5 deg        │                         │                  │
├───────────────────┴─────────────────────────┴──────────────────┤
│  📝 LOGS                                                       │
│  [12:34:56] ✅ VLM solver converged in 45 iterations          │
│  [12:34:55] ℹ️ Solving panel system...                        │
│  [12:34:54] 🌊 Starting VLM calculation for Front Wing        │
└────────────────────────────────────────────────────────────────┘
```

### Flujo de Uso
1. **Seleccionar Método**: VLM o CFD
2. **Configurar**:
   - Component: Front Wing / Rear Wing / Floor / Diffuser
   - NACA Profile: 6412, 4415, 4418, 9618, 0009, 23012
   - Samples: 1-100
   - Velocity: km/h
   - Angle of Attack: grados
3. **Generate**: Click botón ▶ Generate
4. **Monitorear**: Ver progreso, gráficos en tiempo real, logs
5. **Exportar**: Click 📊 Export para descargar JSON

### Datos Generados
```json
{
  "method": "VLM",
  "component": "front_wing",
  "nacaProfile": "NACA6412",
  "samples": 100,
  "results": [
    {
      "velocity": 300,
      "aoa": 5,
      "cl": 1.245,
      "cd": 0.045,
      "pressure": [0.5, 0.3, -0.2, ...],
      "circulation": [...]
    }
  ],
  "statistics": {
    "cl_mean": 1.245,
    "cl_std": 0.123,
    "cd_mean": 0.045,
    "cd_std": 0.008
  }
}
```

### Colores
- **Primary**: #00c8ff (cyan) - VLM mode
- **Secondary**: #00ff88 (green) - Success messages
- **Warning**: #ff8800 (orange) - Warnings
- **Error**: #ff0000 (red) - Errors

---

## 3. QuantumOptimizationDashboard - Optimización Cuántica {#quantumoptimizationdashboard}

### Layout
```
┌─────────────────────────────────────────────────────────────┐
│             ⚛️ Quantum Optimization Dashboard               │
│                  QAOA • VQE • Quantum Annealing             │
├─────────────────────────────────────────────────────────────┤
│  CONFIGURATION                                              │
│  ┌──────────────┬──────────────┬──────────────┐            │
│  │ Method       │ Optimization │ Constraints  │            │
│  │ [QAOA    ▾] │ [Layout  ▾]  │ Flutter: 1.2 │            │
│  │ Iterations:  │ [●] Vibration│ Mass: 50 kg  │            │
│  │ ▰▰▰▰▱ 100   │ [●] Thermal  │              │            │
│  │ Depth: 5     │ [ ] Acoustic │              │            │
│  └──────────────┴──────────────┴──────────────┘            │
│                                                              │
│  [▶️ Run Optimization]  [💾 Export Results]                │
│                                                              │
│  ▰▰▰▰▰▰▰▰▰▱ 90/100 iterations (Est. 15s remaining)        │
├─────────────────────────────────────────────────────────────┤
│  RESULTS                                                     │
│  ┌────────────────────┬────────────────────────────────┐   │
│  │ Convergence Plot   │ Binary Variables Grid          │   │
│  │                    │ ■■□■□■■□□■■■□■□■■□□■         │   │
│  │      📉            │ □■■□■□■■■□□■□■■□■■□■         │   │
│  │                    │ ■□■■□■□□■■□■■□■□■■■□         │   │
│  │                    │ Variables: 200 | Active: 124   │   │
│  └────────────────────┴────────────────────────────────┘   │
│                                                              │
│  Best Energy: -45.23  |  Iterations: 100  |  Depth: 5      │
├─────────────────────────────────────────────────────────────┤
│  📝 QUANTUM CIRCUIT LOGS                                    │
│  [12:45:23] ✅ Optimization converged! Final energy: -45.23│
│  [12:45:22] ⚛️ Iteration 100: E = -45.23, improvement = 0.01│
│  [12:45:21] ⚛️ Iteration 99: E = -45.22, parameters updated │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de Uso
1. **Seleccionar Método**: QAOA / VQE / Quantum Annealing
2. **Elegir Optimización**:
   - Stiffener Layout (posición de rigidizadores)
   - Thickness Distribution (distribución de espesor)
   - Cooling Topology (topología de enfriamiento)
   - Complete Wing (ala completa)
   - Aeroelastic Flutter (optimización de flutter)
3. **Configurar Restricciones**:
   - Flutter Margin: 1.2x mínimo
   - Max Displacement: 0.05 m
   - Max Mass: 50 kg
4. **Toggle Multi-física**: Vibration, Thermal, Aeroacoustic
5. **Run**: Click ▶️ Run Optimization
6. **Ver Resultados**: Convergencia, variables binarias, energía óptima
7. **Exportar**: Click 💾 Export Results

### Algoritmos Cuánticos

#### QAOA (Quantum Approximate Optimization Algorithm)
- Iterations: 50-200
- Depth: 1-10 (número de capas cuánticas)
- Uso: Problemas combinatorios (layout, topology)

#### VQE (Variational Quantum Eigensolver)
- Iterations: 100-500
- Ansatz: Hardware-efficient
- Uso: Optimización continua (thickness, flutter)

#### Quantum Annealing
- Anneal Time: 20 µs
- Qubits: 200+
- Uso: Problemas QUBO grandes

### QUBO Formulation
```
H = Σ w_ij * x_i * x_j + Σ h_i * x_i + λ * (constraints)²
```
- **x_i**: Variables binarias (0 o 1)
- **w_ij**: Pesos de interacción
- **h_i**: Bias de campo
- **λ**: Penalización de restricciones

### Colores
- **Primary**: #8800ff (purple) - Quantum theme
- **Secondary**: #ff00ff (magenta) - Active qubits
- **Energy**: #ff0088 (pink) - Energy values

---

## 4. AdvancedAeroVisualization3D - Visualización 3D {#advancedaerovisualization3d}

### Layout
```
┌─────────────────────────────────────────────────────────────┐
│  [🎨 Pressure] [💨 Streamlines] [⬆️ Forces] [🌀 Vortex]   │
│  [□ Mesh]  Colormap: [Jet ▾]                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                       ┌─────────────┐                       │
│                       │             │                       │
│                      🏎️   3D WING  │                       │
│                       │             │                       │
│                       └─────────────┘                       │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │ Pressure Scale:  [-3 ──────────── +1]           │       │
│  │                  Blue → Cyan → Green → Yellow → Red│       │
│  └──────────────────────────────────────────────────┘       │
├─────────────────────────────────────────────────────────────┤
│  ℹ️ INFO                                                    │
│  Geometry: Front Wing NACA6412                              │
│  Chord: 0.5m  |  Span: 1.8m  |  Points: 800                │
│  Forces: ↓ Downforce: 1200N  → Drag: 150N  ← Side: 50N     │
└─────────────────────────────────────────────────────────────┘
```

### Controles
1. **Mouse**:
   - **Left drag**: Rotar cámara (OrbitControls)
   - **Right drag**: Pan (mover vista)
   - **Scroll**: Zoom in/out
2. **Toggles**:
   - 🎨 **Pressure**: Mostrar distribución de presión con colormap
   - 💨 **Streamlines**: Mostrar líneas de flujo
   - ⬆️ **Forces**: Mostrar vectores de fuerza (arrows 3D)
   - 🌀 **Vortex**: Mostrar indicadores de vórtice (torus)
   - □ **Mesh**: Mostrar wireframe de geometría
3. **Colormap**: Jet (azul→rojo) / Viridis (morado→amarillo)

### Visualizaciones

#### Pressure Distribution
- **Colormap Jet**: 
  - Azul: Baja presión (-3)
  - Cyan: Presión media baja (-1)
  - Verde: Presión neutra (0)
  - Amarillo: Presión media alta (+0.5)
  - Rojo: Alta presión (+1)
- **Mesh**: 40 x 20 puntos (800 triángulos)

#### Streamlines
- **Origen**: Borde de ataque, espaciado uniforme en span
- **Integración**: Método Euler, dt = 0.05
- **Longitud**: 50 steps (2.5 chords)
- **Color**: Gradiente según velocidad

#### Force Vectors
- **Downforce**: ↓ Flecha azul vertical
- **Drag**: → Flecha roja horizontal
- **Sideforce**: ← Flecha verde lateral
- **Escala**: Proporcional a magnitud de fuerza
- **Labels**: Texto 3D con valor en N

#### Vortex Indicators
- **Geometría**: Torus (radio: 0.05m)
- **Color**: Magenta (#ff00ff)
- **Ubicación**: Regiones de alta vorticidad (ωz > 100 s⁻¹)
- **Típicas**: Tip vortex, trailing edge vortex

### Ejemplo de Datos
```javascript
const wingData = {
  geometry: {
    component: 'front_wing',
    nacaProfile: 'NACA6412',
    chord: 0.5,
    span: 1.8,
  },
  pressure: Float32Array[800],  // -3 to +1
  velocity: Float32Array[2400], // [vx, vy, vz] × 800
  forces: {
    downforce: -1200,  // N (negative = downward)
    drag: 150,         // N
    sideforce: 50      // N
  },
  vorticity: Float32Array[800]  // s⁻¹
};
```

### Colores
- **Pressure Jet**: #0000ff → #00ffff → #00ff00 → #ffff00 → #ff0000
- **Pressure Viridis**: #440154 → #31688e → #35b779 → #fde724
- **Streamlines**: Gradiente velocidad
- **Forces**: Downforce=#0088ff, Drag=#ff0000, Sideforce=#00ff88

---

## 5. MultiphysicsRealtimeDashboard - Dashboard Multifísica {#multiphysicsrealtimedashboard}

### Layout
```
┌─────────────────────────────────────────────────────────────┐
│          ⚛️ Dashboard Multifísica en Tiempo Real            │
│      Aeroelástica • Vibración • Térmico • Aeroacústica      │
├─────────────────────────────────────────────────────────────┤
│  [●] 〰️ Aeroelástica  [●] 🌊 Vibración                     │
│  [●] 🔥 Térmico       [●] 🔊 Aeroacústica                   │
│                                                              │
│  Velocidad: [300] km/h    [▶️ Iniciar] [💾 Exportar]       │
│  ▰▰▰▰▰▰▱▱▱▱ Paso 70/100  Tiempo: 7.0s                     │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┬──────────────────────┐           │
│  │ 〰️ AEROELÁSTICO     │ 🌊 VIBRACIÓN         │           │
│  │                      │                      │           │
│  │ Vel. Flutter: 285 km/h│   Aceleración       │           │
│  │ Margen: 1.05 ⚠️      │      📈             │           │
│  │                      │                      │           │
│  │ Desplazamiento:      │   Resonance Peaks:   │           │
│  │     ∿∿∿∿           │   15.2 Hz ▰▰▰▰▰    │           │
│  │                      │   22.5 Hz ▰▰▱▱▱    │           │
│  │ Frecuencias:         │   35.8 Hz ▰▱▱▱▱    │           │
│  │ Modo 1: 15.2 Hz      │                      │           │
│  │ Modo 2: 22.5 Hz      │                      │           │
│  │ Modo 3: 35.8 Hz      │                      │           │
│  └──────────────────────┴──────────────────────┘           │
│  ┌──────────────────────┬──────────────────────┐           │
│  │ 🔥 TÉRMICO          │ 🔊 AEROACÚSTICO      │           │
│  │                      │                      │           │
│  │ Freno: 750°C ▰▰▰▰▰  │ SPL Total: 105 dB    │           │
│  │ Piso:   95°C ▰▰▱▱▱  │                      │           │
│  │ Ala:    55°C ▰▱▱▱▱  │      🔊             │           │
│  │                      │                      │           │
│  │ Flujo de Calor:      │ ✅ FIA Compliant     │           │
│  │     📈              │ (< 110 dB)           │           │
│  │                      │                      │           │
│  │ Eficiencia: 82%      │ Espectro Frecuencia: │           │
│  │                      │ 1kHz: 105 dB ●       │           │
│  │                      │ 2kHz:  82 dB ●       │           │
│  └──────────────────────┴──────────────────────┘           │
├─────────────────────────────────────────────────────────────┤
│  📝 REGISTRO DE CÁLCULOS                                    │
│  [12:50:15] 〰️ Flutter: Vf = 285.3 km/h, Margen = 1.05 ⚠️ │
│  [12:50:14] 🔥 Térmico: T_freno = 750°C, T_piso = 95°C     │
│  [12:50:13] 🔊 Acústica: SPL = 105 dB ✅ FIA Compliant      │
│  [12:50:12] ⚠️ Vibración: Pico detectado 6.5 m/s²          │
└─────────────────────────────────────────────────────────────┘
```

### Módulos de Física

#### 1. Aeroelástica 〰️
**Propósito**: Analizar acoplamiento fluido-estructura y flutter

**Métricas**:
- **Flutter Speed (Vf)**: Velocidad crítica de flutter (km/h)
  - ✅ Safe: Vf > 1.2 × V_operación
  - ⚠️ Warning: 1.0 × V < Vf < 1.2 × V
  - ❌ Critical: Vf < V_operación
- **Flutter Margin**: Ratio Vf / V_operación
- **Modal Frequencies**: Primeras 5 frecuencias naturales (Hz)
- **Damping Ratios**: Amortiguamiento modal (ζ)
- **Displacement**: Desplazamiento en tiempo real (m)

**Visualización**: Gráfico de desplazamiento vs tiempo

**Ecuaciones**:
```
M·ẍ + C·ẋ + K·x = F_aero(V, x, ẋ)
Flutter: det(K - ω²M + iωC - Q_aero) = 0
```

#### 2. Vibración 🌊
**Propósito**: Monitorear vibraciones estructurales

**Métricas**:
- **Acceleration**: Aceleración en m/s²
- **Velocity**: Velocidad en m/s
- **Displacement**: Desplazamiento en m
- **Resonance Peaks**: Picos en FFT con frecuencias dominantes

**Visualización**: 
- Gráfico de aceleración vs tiempo
- Barras de resonancia con amplitudes

**Alertas**:
- ⚠️ Peak > 6 m/s²: Vibración alta
- ❌ Peak > 10 m/s²: Vibración crítica

**Método**: Integración Newmark-β, dt = 0.001s

#### 3. Térmico 🔥
**Propósito**: Analizar transferencia de calor y temperaturas

**Métricas**:
- **Temperatures**: Por componente (°C)
  - Freno Delantero: Límite 1000°C
  - Freno Trasero: Límite 1000°C
  - Piso: Límite 150°C
  - Ala Delantera: Límite 200°C
  - Difusor: Límite 200°C
- **Heat Flux**: Flujo de calor (W/m²)
- **Thermal Stress**: Estrés térmico (MPa)
- **Cooling Efficiency**: Eficiencia de enfriamiento (%)

**Visualización**:
- Barras de temperatura con límites
- Gráfico de flujo de calor vs tiempo

**Ecuaciones**:
```
ρcp·∂T/∂t = k·∇²T + Q_gen - Q_conv
σ_thermal = E·α·ΔT / (1-ν)
```

#### 4. Aeroacústico 🔊
**Propósito**: Calcular ruido aerodinámico y cumplimiento FIA

**Métricas**:
- **SPL Total**: Sound Pressure Level total (dB)
  - ✅ FIA Compliant: SPL < 110 dB
  - ❌ Non-compliant: SPL ≥ 110 dB
- **Spectrum**: Espectro de frecuencia (dB vs Hz)
  - 100 Hz, 500 Hz, 1 kHz, 2 kHz, 5 kHz

**Visualización**:
- Gráfico SPL vs tiempo con línea de límite FIA
- Scatter plot del espectro

**Método**: Ffowcs Williams-Hawkings (FW-H)

**Ecuación**:
```
p'(x,t) = ∫ [ρ₀(∂vₙ/∂t)/r] dS + ∫ [(∂pₙ)/∂t]/r dS
SPL = 20·log₁₀(p_rms / p_ref),  p_ref = 20 µPa
```

### Simulación en Tiempo Real

**Parámetros**:
- **Velocity**: 50-400 km/h (configurable)
- **Update Interval**: 100 ms (10 Hz)
- **Simulation Time**: 10 segundos (100 pasos)
- **Time Step**: 0.1s

**Algoritmo**:
```javascript
for (step = 0; step < totalSteps; step++) {
  time = step * dt;
  
  // Aeroelástica
  flutterSpeed = calculateFlutter(velocity, geometry);
  modalFreqs = eigenAnalysis(M, K);
  displacement = newmarkBeta(M, C, K, F_aero, dt);
  
  // Vibración
  acceleration = M⁻¹ · (F_external - C·v - K·x);
  fft = fourierTransform(acceleration);
  resonancePeaks = findPeaks(fft);
  
  // Térmico
  temperature = heatEquation(k, Q_gen, Q_conv, dt);
  thermalStress = E·α·ΔT / (1-ν);
  
  // Aeroacústico
  spl = fwh(velocity, geometry, surfacePressure);
  spectrum = frequencyAnalysis(spl);
  
  updateVisualizations();
  logResults();
}
```

### Colores por Módulo
- **Aeroelástica**: #00c8ff (cyan)
- **Vibración**: #00ff88 (green)
- **Térmico**: #ff8800 (orange)
- **Aeroacústico**: #ff00ff (magenta)

### Exportación de Datos
```json
{
  "config": {
    "velocity": 300,
    "updateInterval": 1000,
    "simulationTime": 10
  },
  "results": {
    "aeroelastic": {
      "flutterSpeed": 285.3,
      "flutterMargin": 1.05,
      "modalFrequencies": [15.2, 22.5, 35.8, 48.3, 62.1],
      "displacement": [{ time: 0, value: 0 }, ...]
    },
    "vibration": {
      "acceleration": [...],
      "resonancePeaks": [
        { frequency: 15.2, amplitude: 5.2 },
        { frequency: 22.5, amplitude: 2.8 }
      ]
    },
    "thermal": {
      "temperatures": [
        { component: "Freno", temp: 750, limit: 1000 }
      ],
      "coolingEfficiency": 82
    },
    "aeroacoustic": {
      "totalNoise": 105,
      "fiaCompliant": true,
      "spectrum": [
        { frequency: 1000, spl: 105 }
      ]
    }
  },
  "logs": [...]
}
```

---

## 🎯 Casos de Uso

### Caso 1: Diseño Inicial de Front Wing
1. **AerodynamicDataGenerator**: Generar 50 muestras VLM con NACA6412
2. **AdvancedAeroVisualization3D**: Visualizar distribución de presión
3. **MultiphysicsRealtimeDashboard**: Verificar flutter margin > 1.2
4. **QuantumOptimizationDashboard**: Optimizar layout de rigidizadores

### Caso 2: Análisis de Rear Wing
1. **AerodynamicDataGenerator**: Generar CFD con NACA9618
2. **AdvancedAeroVisualization3D**: Ver streamlines y tip vortex
3. **MultiphysicsRealtimeDashboard**: Monitorear temperatura y vibración
4. **QuantumOptimizationDashboard**: Optimizar thickness distribution

### Caso 3: Optimización de Cooling
1. **MultiphysicsRealtimeDashboard**: Simular térmico, detectar hot spots
2. **QuantumOptimizationDashboard**: Ejecutar cooling topology optimization
3. **AdvancedAeroVisualization3D**: Verificar impacto aerodinámico
4. **AerodynamicDataGenerator**: Re-generar datos con nueva geometría

### Caso 4: Validación FIA
1. **AerodynamicDataGenerator**: Generar datos a velocidad de carrera
2. **MultiphysicsRealtimeDashboard**: Verificar SPL < 110 dB
3. **AdvancedAeroVisualization3D**: Identificar fuentes de ruido (vortex)
4. **QuantumOptimizationDashboard**: Optimizar para reducir ruido

---

## 📱 Shortcuts y Atajos

### Teclado (3D Visualization)
- **R**: Reset cámara
- **P**: Toggle presión
- **S**: Toggle streamlines
- **F**: Toggle fuerzas
- **V**: Toggle vórtices
- **M**: Toggle mesh
- **C**: Cambiar colormap

### Mouse (3D Visualization)
- **Left Drag**: Rotar
- **Right Drag**: Pan
- **Scroll**: Zoom
- **Double Click**: Focus en punto

### Navegación
- **Tab**: Siguiente pestaña
- **Shift+Tab**: Pestaña anterior
- **Ctrl+R**: Refresh stats
- **Ctrl+E**: Export data

---

*Guía Visual Completa - Quantum Aero F1 Prototype*  
*Todos los componentes listos para producción*
