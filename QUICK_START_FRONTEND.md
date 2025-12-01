# 🚀 Guía de Inicio Rápido - Frontend Quantum Aero F1

## 📦 Instalación

### 1. Instalar dependencias
```bash
cd /workspaces/Quantum-Aero-F1-Prototype/frontend
npm install
```

Todas las dependencias ya están en `package.json`:
- ✅ react, react-dom
- ✅ @react-three/fiber, @react-three/drei, three
- ✅ axios
- ✅ recharts
- ✅ tailwindcss, @headlessui/react, @heroicons/react

---

## 🔧 Configuración del App.jsx

### Opción Recomendada: Usar QuantumAeroApp

Editar `/frontend/src/App.jsx`:

```javascript
import React from 'react';
import QuantumAeroApp from './components/QuantumAeroApp';

function App() {
  return <QuantumAeroApp />;
}

export default App;
```

---

## 🎯 Estructura de Componentes

```
QuantumAeroApp (main)
├── Tab 1: AerodynamicDataGenerator
│   ├── CFD/VLM dual mode
│   ├── NACA profiles selector
│   ├── Real-time charts
│   └── Export to JSON
├── Tab 2: QuantumOptimizationDashboard
│   ├── QAOA/VQE/Annealing
│   ├── 5 optimization types
│   ├── Binary variables grid
│   └── Convergence plots
├── Tab 3: AdvancedAeroVisualization3D
│   ├── Pressure distribution (3D)
│   ├── Streamlines
│   ├── Force vectors
│   └── Vortex indicators
└── Tab 4: MultiphysicsRealtimeDashboard
    ├── Aeroelastic analysis
    ├── Vibration monitoring
    ├── Thermal analysis
    └── Aeroacoustic SPL
```

---

## 🖥️ Backend Requirements

### 1. VLM Solver (required for AerodynamicDataGenerator)

**Endpoint**: `http://localhost:8001/vlm/solve`

**Start backend**:
```bash
cd /workspaces/Quantum-Aero-F1-Prototype
python realtime_server.py
```

o si existe servicio específico:
```bash
cd services/physics_engine
python vlm_server.py
```

**Test endpoint**:
```bash
curl -X POST http://localhost:8001/vlm/solve \
  -H "Content-Type: application/json" \
  -d '{
    "geometry": {
      "component": "front_wing",
      "nacaProfile": "NACA6412",
      "chord": 0.5,
      "span": 1.8
    },
    "flowConditions": {
      "velocity": 300,
      "angleOfAttack": 5,
      "rho": 1.225
    }
  }'
```

### 2. Quantum Service (optional, simulated by default)

**Endpoint**: `http://localhost:8002/quantum/optimize`

Si no existe, el componente usa simulación local.

---

## ▶️ Ejecutar la Aplicación

### Development Mode
```bash
cd /workspaces/Quantum-Aero-F1-Prototype/frontend
npm start
```

La app se abrirá en: `http://localhost:3000`

### Production Build
```bash
npm run build
```

Output en: `/frontend/build/`

---

## 🎨 Uso de Componentes Individuales

### Importar solo un componente

```javascript
import React from 'react';
import AerodynamicDataGenerator from './components/AerodynamicDataGenerator';
import './components/AerodynamicDataGenerator.css';

function App() {
  return (
    <div className="App">
      <AerodynamicDataGenerator />
    </div>
  );
}
```

### Con callback de datos guardados

```javascript
import React from 'react';
import AerodynamicDataGenerator from './components/AerodynamicDataGenerator';
import { useAeroDataStorage } from './utils/AeroDataStorage';

function App() {
  const storage = useAeroDataStorage();

  const handleDataSaved = async () => {
    const stats = await storage.getStorageStats();
    console.log('Datos guardados:', stats);
  };

  return <AerodynamicDataGenerator onDataSaved={handleDataSaved} />;
}
```

---

## 💾 IndexedDB Storage

Los datos se guardan automáticamente en IndexedDB del navegador.

### Ver datos en DevTools
1. Abrir Chrome DevTools (F12)
2. Ir a "Application" tab
3. Expandir "IndexedDB"
4. Ver base de datos "AeroDataDB"

### Limpiar datos
```javascript
// En consola del navegador
const request = indexedDB.deleteDatabase('AeroDataDB');
request.onsuccess = () => console.log('Database deleted');
```

---

## 🧪 Testing

### Ejecutar tests
```bash
npm test
```

### Test de componente individual
```bash
npm test -- AerodynamicDataGenerator.test.jsx
```

---

## 📊 Funcionalidades por Componente

### 1. AerodynamicDataGenerator
- ✅ Generar 1-100 muestras VLM/CFD
- ✅ Seleccionar componente F1 (Front Wing, Rear Wing, Floor, Diffuser)
- ✅ Elegir perfil NACA (6412, 4415, 4418, 9618, 0009, 23012)
- ✅ Configurar condiciones de flujo (velocidad, AoA, densidad)
- ✅ Ver gráfico de presión en tiempo real
- ✅ Estadísticas (CL, CD mean/std)
- ✅ Logs con timestamps
- ✅ Exportar a JSON

### 2. QuantumOptimizationDashboard
- ✅ Seleccionar método cuántico (QAOA, VQE, Annealing)
- ✅ Elegir tipo de optimización (5 opciones)
- ✅ Configurar restricciones (flutter, displacement, mass)
- ✅ Toggle multi-física (vibration, thermal, aeroacoustic)
- ✅ Ver convergencia de energía
- ✅ Interactuar con grid de variables binarias
- ✅ Logs de circuito cuántico
- ✅ Exportar resultados

### 3. AdvancedAeroVisualization3D
- ✅ Cargar datos de VLM/CFD
- ✅ Rotar/zoom con mouse (OrbitControls)
- ✅ Toggle presión/streamlines/fuerzas/vórtices/mesh
- ✅ Cambiar esquema de color (jet/viridis)
- ✅ Ver leyenda de presión
- ✅ Info panel con geometría y fuerzas

### 4. MultiphysicsRealtimeDashboard
- ✅ Iniciar/detener simulación en tiempo real
- ✅ Configurar velocidad (km/h)
- ✅ Toggle módulos de física
- ✅ Ver flutter speed y margen
- ✅ Monitorear vibración (aceleración, velocidad, desplazamiento)
- ✅ Ver temperaturas por componente
- ✅ SPL aeroacústico con límite FIA
- ✅ Exportar datos completos

---

## 🔗 Integración con Backend Real

### Modificar URL de API

En `AerodynamicDataGenerator.jsx`, línea ~150:

```javascript
// Cambiar de:
const response = await axios.post('http://localhost:8001/vlm/solve', {...});

// A tu backend:
const response = await axios.post('http://your-backend.com/api/vlm/solve', {...});
```

### Variables de entorno

Crear `.env` en `/frontend/`:

```env
REACT_APP_VLM_API=http://localhost:8001
REACT_APP_QUANTUM_API=http://localhost:8002
REACT_APP_PHYSICS_API=http://localhost:8003
```

Usar en componentes:
```javascript
const VLM_API = process.env.REACT_APP_VLM_API || 'http://localhost:8001';
```

---

## 🐛 Troubleshooting

### Error: "Cannot find module 'axios'"
```bash
npm install axios
```

### Error: "Cannot find module '@react-three/fiber'"
```bash
npm install @react-three/fiber @react-three/drei three
```

### Error: "Cannot find module 'recharts'"
```bash
npm install recharts
```

### CORS Error al llamar backend
En backend (Python Flask/FastAPI), agregar:
```python
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Habilitar CORS
```

### IndexedDB no funciona
- Verificar que el navegador soporte IndexedDB (todos modernos sí)
- Verificar que no esté en modo incógnito
- Limpiar cache del navegador

### 3D visualization no renderiza
- Verificar que WebGL esté habilitado en navegador
- Probar en Chrome/Firefox actualizado
- Ver errores en consola (F12)

---

## 📱 Responsive Design

Todos los componentes son responsive:

- **Desktop** (>1200px): Grid completo, 2-3 columnas
- **Tablet** (768-1200px): 1-2 columnas, navegación adaptativa
- **Mobile** (<768px): 1 columna, controles apilados

---

## ⚡ Performance Tips

### 1. React.memo para componentes pesados
```javascript
export default React.memo(AdvancedAeroVisualization3D);
```

### 2. useMemo para cálculos costosos
```javascript
const processedData = React.useMemo(() => {
  return expensiveCalculation(data);
}, [data]);
```

### 3. useCallback para funciones
```javascript
const handleDataSaved = React.useCallback(() => {
  updateStats();
}, []);
```

### 4. Lazy loading de componentes
```javascript
const MultiphysicsRealtimeDashboard = React.lazy(() => 
  import('./components/MultiphysicsRealtimeDashboard')
);
```

---

## 📖 Documentación Adicional

- **Three.js Docs**: https://threejs.org/docs/
- **React Three Fiber**: https://docs.pmnd.rs/react-three-fiber
- **Recharts**: https://recharts.org/en-US/
- **IndexedDB API**: https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API

---

## ✅ Checklist de Implementación

- [x] Instalar dependencias
- [x] Crear todos los componentes
- [x] Crear archivos CSS
- [x] Integrar en App.jsx
- [ ] Configurar backend VLM
- [ ] Probar conexión API
- [ ] Verificar almacenamiento IndexedDB
- [ ] Test en diferentes navegadores
- [ ] Build de producción
- [ ] Deploy

---

## 🎯 Next Steps

1. **Ejecutar frontend**: `npm start`
2. **Ejecutar backend VLM**: `python realtime_server.py`
3. **Abrir navegador**: `http://localhost:3000`
4. **Probar cada tab**:
   - Generar datos VLM
   - Ejecutar optimización cuántica
   - Visualizar en 3D
   - Simular multifísica
5. **Verificar IndexedDB** en DevTools

---

*Última actualización: Diciembre 2024*  
*Quantum Aero F1 Prototype - Quick Start Guide*
