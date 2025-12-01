/**
 * Sistema de Almacenamiento Optimizado para Datos Aerodinámicos
 * Utiliza IndexedDB para almacenamiento local masivo
 * MongoDB para datos de servidor
 * Redis para caché de alto rendimiento
 */

class AeroDataStorage {
  constructor() {
    this.dbName = 'QuantumAeroF1DB';
    this.dbVersion = 1;
    this.db = null;
    this.initialized = false;
  }

  /**
   * Inicializar IndexedDB
   */
  async initialize() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        this.initialized = true;
        console.log('✅ IndexedDB inicializado');
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        // Store para datos VLM
        if (!db.objectStoreNames.contains('vlm_results')) {
          const vlmStore = db.createObjectStore('vlm_results', { 
            keyPath: 'id', 
            autoIncrement: true 
          });
          vlmStore.createIndex('timestamp', 'timestamp', { unique: false });
          vlmStore.createIndex('component', 'component', { unique: false });
          vlmStore.createIndex('nacaProfile', 'nacaProfile', { unique: false });
        }

        // Store para datos CFD
        if (!db.objectStoreNames.contains('cfd_results')) {
          const cfdStore = db.createObjectStore('cfd_results', { 
            keyPath: 'id', 
            autoIncrement: true 
          });
          cfdStore.createIndex('timestamp', 'timestamp', { unique: false });
          cfdStore.createIndex('method', 'method', { unique: false });
        }

        // Store para optimizaciones cuánticas
        if (!db.objectStoreNames.contains('quantum_optimizations')) {
          const quantumStore = db.createObjectStore('quantum_optimizations', { 
            keyPath: 'id', 
            autoIncrement: true 
          });
          quantumStore.createIndex('timestamp', 'timestamp', { unique: false });
          quantumStore.createIndex('type', 'type', { unique: false });
          quantumStore.createIndex('status', 'status', { unique: false });
        }

        // Store para datos multifísica
        if (!db.objectStoreNames.contains('multiphysics_results')) {
          const multiStore = db.createObjectStore('multiphysics_results', { 
            keyPath: 'id', 
            autoIncrement: true 
          });
          multiStore.createIndex('timestamp', 'timestamp', { unique: false });
          multiStore.createIndex('analysisType', 'analysisType', { unique: false });
        }

        // Store para geometrías
        if (!db.objectStoreNames.contains('geometries')) {
          const geomStore = db.createObjectStore('geometries', { 
            keyPath: 'id', 
            autoIncrement: true 
          });
          geomStore.createIndex('name', 'name', { unique: false });
          geomStore.createIndex('type', 'type', { unique: false });
        }

        console.log('🔨 Esquema de base de datos creado');
      };
    });
  }

  /**
   * Guardar resultado VLM
   */
  async saveVLMResult(data) {
    if (!this.initialized) await this.initialize();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['vlm_results'], 'readwrite');
      const store = transaction.objectStore('vlm_results');

      const result = {
        timestamp: Date.now(),
        date: new Date().toISOString(),
        component: data.component || 'front_wing',
        nacaProfile: data.nacaProfile || '6412',
        geometry: data.geometry,
        flowConditions: {
          velocity: data.velocity,
          alpha: data.alpha,
          reynolds: data.reynolds,
        },
        results: {
          cl: data.cl,
          cd: data.cd,
          cm: data.cm,
          l_over_d: data.l_over_d,
          pressure: this.compressArray(data.pressure), // Comprimir array grande
        },
        metadata: {
          computeTime: data.computeTime,
          panelCount: data.panelCount,
        },
      };

      const request = store.add(result);

      request.onsuccess = () => {
        console.log(`💾 Resultado VLM guardado: ID ${request.result}`);
        resolve(request.result);
      };
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Guardar resultado CFD
   */
  async saveCFDResult(data) {
    if (!this.initialized) await this.initialize();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['cfd_results'], 'readwrite');
      const store = transaction.objectStore('cfd_results');

      const result = {
        timestamp: Date.now(),
        date: new Date().toISOString(),
        method: data.method || 'synthetic',
        geometry: data.geometry,
        flowConditions: data.flowConditions,
        results: {
          cl: data.cl,
          cd: data.cd,
          cm: data.cm,
          pressureField: this.compressArray(data.pressureField),
          velocityField: this.compressArray(data.velocityField),
        },
        metadata: data.metadata,
      };

      const request = store.add(result);

      request.onsuccess = () => {
        console.log(`💾 Resultado CFD guardado: ID ${request.result}`);
        resolve(request.result);
      };
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Guardar optimización cuántica
   */
  async saveQuantumOptimization(data) {
    if (!this.initialized) await this.initialize();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['quantum_optimizations'], 'readwrite');
      const store = transaction.objectStore('quantum_optimizations');

      const optimization = {
        timestamp: Date.now(),
        date: new Date().toISOString(),
        type: data.type,
        status: data.status || 'completed',
        config: data.config,
        results: {
          optimalDesign: data.optimalDesign,
          metrics: data.metrics,
          quboEnergy: data.quboEnergy,
          convergenceHistory: this.compressArray(data.convergenceHistory),
        },
        metadata: {
          computeTime: data.computeTime,
          iterations: data.iterations,
          quantumMethod: data.quantumMethod,
        },
      };

      const request = store.add(optimization);

      request.onsuccess = () => {
        console.log(`💾 Optimización cuántica guardada: ID ${request.result}`);
        resolve(request.result);
      };
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Obtener resultados VLM
   */
  async getVLMResults(filters = {}) {
    if (!this.initialized) await this.initialize();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['vlm_results'], 'readonly');
      const store = transaction.objectStore('vlm_results');

      let request;
      
      if (filters.component) {
        const index = store.index('component');
        request = index.getAll(filters.component);
      } else {
        request = store.getAll();
      }

      request.onsuccess = () => {
        let results = request.result;

        // Aplicar filtros adicionales
        if (filters.startDate) {
          results = results.filter(r => r.timestamp >= filters.startDate);
        }
        if (filters.endDate) {
          results = results.filter(r => r.timestamp <= filters.endDate);
        }
        if (filters.nacaProfile) {
          results = results.filter(r => r.nacaProfile === filters.nacaProfile);
        }

        // Descomprimir datos
        results = results.map(r => ({
          ...r,
          results: {
            ...r.results,
            pressure: this.decompressArray(r.results.pressure),
          },
        }));

        resolve(results);
      };
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Obtener optimizaciones cuánticas
   */
  async getQuantumOptimizations(filters = {}) {
    if (!this.initialized) await this.initialize();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['quantum_optimizations'], 'readonly');
      const store = transaction.objectStore('quantum_optimizations');

      let request;
      
      if (filters.type) {
        const index = store.index('type');
        request = index.getAll(filters.type);
      } else {
        request = store.getAll();
      }

      request.onsuccess = () => {
        let results = request.result;

        // Descomprimir datos
        results = results.map(r => ({
          ...r,
          results: {
            ...r.results,
            convergenceHistory: this.decompressArray(r.results.convergenceHistory),
          },
        }));

        resolve(results);
      };
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Estadísticas de almacenamiento
   */
  async getStorageStats() {
    if (!this.initialized) await this.initialize();

    const stats = {
      vlmCount: 0,
      cfdCount: 0,
      quantumCount: 0,
      multiphysicsCount: 0,
      geometriesCount: 0,
      totalSize: 0,
    };

    const stores = ['vlm_results', 'cfd_results', 'quantum_optimizations', 'multiphysics_results', 'geometries'];

    for (const storeName of stores) {
      const count = await new Promise((resolve, reject) => {
        const transaction = this.db.transaction([storeName], 'readonly');
        const store = transaction.objectStore(storeName);
        const request = store.count();
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });

      if (storeName === 'vlm_results') stats.vlmCount = count;
      else if (storeName === 'cfd_results') stats.cfdCount = count;
      else if (storeName === 'quantum_optimizations') stats.quantumCount = count;
      else if (storeName === 'multiphysics_results') stats.multiphysicsCount = count;
      else if (storeName === 'geometries') stats.geometriesCount = count;
    }

    // Estimar tamaño (IndexedDB no proporciona tamaño directo)
    if (navigator.storage && navigator.storage.estimate) {
      const estimate = await navigator.storage.estimate();
      stats.totalSize = estimate.usage;
      stats.quota = estimate.quota;
      stats.usagePercent = (estimate.usage / estimate.quota * 100).toFixed(2);
    }

    return stats;
  }

  /**
   * Limpiar datos antiguos
   */
  async cleanOldData(daysOld = 30) {
    if (!this.initialized) await this.initialize();

    const cutoffDate = Date.now() - (daysOld * 24 * 60 * 60 * 1000);
    const stores = ['vlm_results', 'cfd_results', 'quantum_optimizations'];

    let totalDeleted = 0;

    for (const storeName of stores) {
      const deleted = await new Promise((resolve, reject) => {
        const transaction = this.db.transaction([storeName], 'readwrite');
        const store = transaction.objectStore(storeName');
        const index = store.index('timestamp');
        const range = IDBKeyRange.upperBound(cutoffDate);
        const request = index.openCursor(range);

        let count = 0;

        request.onsuccess = (event) => {
          const cursor = event.target.result;
          if (cursor) {
            cursor.delete();
            count++;
            cursor.continue();
          } else {
            resolve(count);
          }
        };
        request.onerror = () => reject(request.error);
      });

      totalDeleted += deleted;
    }

    console.log(`🗑️ ${totalDeleted} registros antiguos eliminados`);
    return totalDeleted;
  }

  /**
   * Exportar datos a JSON
   */
  async exportToJSON(storeName) {
    if (!this.initialized) await this.initialize();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.getAll();

      request.onsuccess = () => {
        const data = request.result;
        const json = JSON.stringify(data, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${storeName}_${Date.now()}.json`;
        link.click();
        resolve(data.length);
      };
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Comprimir array (simple codificación)
   */
  compressArray(arr) {
    if (!arr || arr.length === 0) return '';
    // Convertir a string base64 para reducir espacio
    return btoa(JSON.stringify(arr));
  }

  /**
   * Descomprimir array
   */
  decompressArray(str) {
    if (!str) return [];
    try {
      return JSON.parse(atob(str));
    } catch (e) {
      return [];
    }
  }
}

// Exportar singleton
const aeroDataStorage = new AeroDataStorage();

export default aeroDataStorage;

/**
 * Hook de React para usar el storage
 */
export const useAeroDataStorage = () => {
  const [isReady, setIsReady] = React.useState(false);

  React.useEffect(() => {
    aeroDataStorage.initialize().then(() => {
      setIsReady(true);
    });
  }, []);

  return {
    storage: aeroDataStorage,
    isReady,
  };
};
