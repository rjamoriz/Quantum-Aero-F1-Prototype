/**
 * High-Speed Aerodynamic Load Generator Utility
 * Generates synthetic aerodynamic loads for F1 components under high-speed conditions
 * 
 * Features:
 * - Thin airfoil theory for pressure distributions
 * - Panel method approximations
 * - Vortex Lattice Method (VLM) simplified
 * - Time-varying transient loads
 * - Export to standard formats (JSON, CSV, HDF5-compatible)
 */

/**
 * Thin Airfoil Theory - Generates pressure distribution along chord
 * @param {Object} config - Configuration object
 * @param {string} config.profile - NACA 4-digit profile (e.g., '2412')
 * @param {number} config.velocity - Freestream velocity (km/h)
 * @param {number} config.angleOfAttack - Angle of attack (degrees)
 * @param {number} config.chord - Chord length (m)
 * @param {number} config.span - Span length (m)
 * @param {number} config.numPoints - Number of chord points (default: 100)
 * @returns {Object} Aerodynamic loads with pressure distribution
 */
export const generateThinAirfoilLoads = (config) => {
  const {
    profile = '2412',
    velocity = 300,
    angleOfAttack = 5,
    chord = 0.5,
    span = 1.8,
    numPoints = 100,
  } = config;

  // Convert velocity to m/s
  const V = velocity / 3.6;
  
  // Air properties
  const rho = 1.225; // kg/m³
  const mu = 1.81e-5; // Pa·s
  
  // Dynamic pressure
  const q = 0.5 * rho * V * V;
  
  // Reynolds number
  const Re = (rho * V * chord) / mu;
  
  // Parse NACA profile
  const m = parseInt(profile[0]) / 100; // Maximum camber
  const p = parseInt(profile[1]) / 10;  // Position of max camber
  const t = parseInt(profile.substr(2)) / 100; // Thickness
  
  // Angle of attack in radians
  const alpha = angleOfAttack * Math.PI / 180;
  
  // Generate pressure distribution
  const pressureDistribution = [];
  let totalLift = 0;
  let totalDrag = 0;
  
  for (let i = 0; i < numPoints; i++) {
    const x = i / (numPoints - 1); // Normalized chord position [0, 1]
    
    // Camber line
    let yc = 0;
    let dyc_dx = 0;
    
    if (x < p && p > 0) {
      yc = (m / (p * p)) * (2 * p * x - x * x);
      dyc_dx = (2 * m / (p * p)) * (p - x);
    } else if (x >= p && p > 0) {
      yc = (m / ((1 - p) * (1 - p))) * ((1 - 2 * p) + 2 * p * x - x * x);
      dyc_dx = (2 * m / ((1 - p) * (1 - p))) * (p - x);
    }
    
    // Thickness distribution (NACA 4-digit)
    const yt = 5 * t * (
      0.2969 * Math.sqrt(x) -
      0.1260 * x -
      0.3516 * x * x +
      0.2843 * x * x * x -
      0.1015 * x * x * x * x
    );
    
    // Thin airfoil theory for pressure coefficient
    // Upper surface
    let cp_upper = 0;
    if (x > 0 && x < 1) {
      // Simplified pressure coefficient with camber and angle of attack
      const beta = Math.atan(dyc_dx);
      const alpha_eff = alpha + beta;
      
      // Pressure coefficient from thin airfoil theory
      cp_upper = -2 * alpha_eff * Math.sqrt((1 - x) / x) - 0.5 * (yt / chord);
    }
    
    // Lower surface
    let cp_lower = 0;
    if (x > 0 && x < 1) {
      const beta = Math.atan(dyc_dx);
      const alpha_eff = alpha + beta;
      cp_lower = 2 * alpha_eff * Math.sqrt((1 - x) / x) + 0.2 * (yt / chord);
    }
    
    // Pressure values
    const p_upper = -cp_upper * q;
    const p_lower = -cp_lower * q;
    
    // Normal force per unit chord
    const dx = chord / (numPoints - 1);
    const dF_n = (p_lower - p_upper) * dx;
    
    totalLift += dF_n * Math.cos(alpha);
    totalDrag += dF_n * Math.sin(alpha);
    
    pressureDistribution.push({
      x: x * chord,
      x_normalized: x,
      cp_upper,
      cp_lower,
      p_upper,
      p_lower,
      yc: yc * chord,
      yt_upper: (yc + yt) * chord,
      yt_lower: (yc - yt) * chord,
    });
  }
  
  // Total forces (per unit span)
  const L_per_span = totalLift;
  const D_per_span = totalDrag;
  
  // Total forces
  const L = L_per_span * span;
  const D = D_per_span * span;
  
  // Lift and drag coefficients
  const S = chord * span;
  const cl = L / (q * S);
  const cd = D / (q * S);
  
  // Theoretical lift coefficient (thin airfoil theory)
  const cl_theory = 2 * Math.PI * alpha;
  
  // Pitching moment (about quarter chord)
  const cm = -Math.PI * alpha / 2;
  
  return {
    config,
    aerodynamics: {
      velocity: V,
      velocityKmh: velocity,
      dynamicPressure: q,
      reynoldsNumber: Re,
      alpha: angleOfAttack,
      alphaRad: alpha,
    },
    forces: {
      lift: L,
      drag: D,
      liftPerSpan: L_per_span,
      dragPerSpan: D_per_span,
    },
    coefficients: {
      cl,
      cd,
      cm,
      cl_theory,
      liftSlope: 2 * Math.PI,
    },
    distribution: pressureDistribution,
    metadata: {
      method: 'thin_airfoil_theory',
      timestamp: new Date().toISOString(),
    },
  };
};

/**
 * Panel Method - Simplified panel method for more accurate loads
 * @param {Object} config - Configuration object
 * @returns {Object} Aerodynamic loads with panel-based pressure
 */
export const generatePanelMethodLoads = (config) => {
  const {
    profile = '2412',
    velocity = 300,
    angleOfAttack = 5,
    chord = 0.5,
    span = 1.8,
    numPanels = 50,
  } = config;

  // Use thin airfoil as base and apply panel corrections
  const thinAirfoilResult = generateThinAirfoilLoads({
    ...config,
    numPoints: numPanels,
  });
  
  // Apply viscous corrections
  const cd_viscous = 0.01 + 0.05 * (angleOfAttack / 10) * (angleOfAttack / 10);
  
  const V = velocity / 3.6;
  const rho = 1.225;
  const q = 0.5 * rho * V * V;
  const S = chord * span;
  
  const D_viscous = cd_viscous * q * S;
  
  return {
    ...thinAirfoilResult,
    forces: {
      ...thinAirfoilResult.forces,
      drag: thinAirfoilResult.forces.drag + D_viscous,
      dragViscous: D_viscous,
      dragInduced: thinAirfoilResult.forces.drag,
    },
    coefficients: {
      ...thinAirfoilResult.coefficients,
      cd: thinAirfoilResult.coefficients.cd + cd_viscous,
      cd_viscous,
      cd_induced: thinAirfoilResult.coefficients.cd,
    },
    metadata: {
      method: 'panel_method_simplified',
      timestamp: new Date().toISOString(),
    },
  };
};

/**
 * Vortex Lattice Method (VLM) - Simplified 3D wing analysis
 * @param {Object} config - Configuration object
 * @returns {Object} 3D aerodynamic loads with spanwise distribution
 */
export const generateVLMLoads = (config) => {
  const {
    velocity = 300,
    angleOfAttack = 5,
    chord = 0.5,
    span = 1.8,
    aspectRatio = null,
    numSpanPanels = 20,
  } = config;

  const AR = aspectRatio || (span * span) / (chord * span);
  const V = velocity / 3.6;
  const rho = 1.225;
  const q = 0.5 * rho * V * V;
  const S = chord * span;
  const alpha = angleOfAttack * Math.PI / 180;
  
  // Prandtl lifting line theory
  const cl_alpha = (2 * Math.PI) / (1 + 2 / AR);
  const cl = cl_alpha * alpha;
  
  // Induced drag coefficient
  const e = 0.85; // Oswald efficiency factor
  const cd_i = (cl * cl) / (Math.PI * e * AR);
  
  // Profile drag
  const cd_0 = 0.008;
  const cd = cd_0 + cd_i;
  
  // Forces
  const L = cl * q * S;
  const D = cd * q * S;
  
  // Spanwise load distribution (elliptical)
  const spanwiseDistribution = [];
  for (let i = 0; i < numSpanPanels; i++) {
    const y = (i / (numSpanPanels - 1) - 0.5) * span;
    const y_normalized = y / (span / 2);
    
    // Elliptical distribution
    const loadFactor = Math.sqrt(1 - y_normalized * y_normalized);
    const localCl = cl * loadFactor * (4 / Math.PI);
    const localLift = localCl * q * chord * (span / numSpanPanels);
    
    spanwiseDistribution.push({
      y,
      y_normalized,
      cl_local: localCl,
      lift_local: localLift,
      circulation: localLift / (rho * V),
    });
  }
  
  return {
    config,
    aerodynamics: {
      velocity: V,
      velocityKmh: velocity,
      dynamicPressure: q,
      alpha: angleOfAttack,
      aspectRatio: AR,
    },
    forces: {
      lift: L,
      drag: D,
    },
    coefficients: {
      cl,
      cd,
      cd_induced: cd_i,
      cd_profile: cd_0,
      cl_alpha: cl_alpha * (180 / Math.PI), // per degree
      oswaldEfficiency: e,
    },
    spanwiseDistribution,
    metadata: {
      method: 'vortex_lattice_method_simplified',
      timestamp: new Date().toISOString(),
    },
  };
};

/**
 * Generate time-varying transient loads
 * @param {Object} config - Configuration object
 * @param {number} config.duration - Duration in seconds
 * @param {number} config.samplingRate - Sampling rate in Hz
 * @param {Array} config.frequencies - Array of excitation frequencies
 * @returns {Object} Time-series aerodynamic loads
 */
export const generateTransientLoads = (config) => {
  const {
    baseVelocity = 300,
    duration = 10,
    samplingRate = 100,
    chord = 0.5,
    span = 1.8,
    turbulenceIntensity = 0.05,
    vortexSheddingFreq = null,
  } = config;

  const dt = 1 / samplingRate;
  const numSteps = Math.floor(duration / dt);
  const rho = 1.225;
  
  const timeSeries = [];
  
  for (let i = 0; i < numSteps; i++) {
    const t = i * dt;
    
    // Base velocity with turbulent fluctuations
    const turbulentFluctuation = turbulenceIntensity * (Math.random() - 0.5) * 2;
    const V_kmh = baseVelocity * (1 + turbulentFluctuation);
    const V = V_kmh / 3.6;
    
    // Vortex shedding (if specified)
    let vortexComponent = 0;
    if (vortexSheddingFreq) {
      const omega = 2 * Math.PI * vortexSheddingFreq;
      vortexComponent = 0.1 * Math.sin(omega * t);
    }
    
    // Angle of attack variation
    const alpha_base = 5;
    const alpha_variation = 2 * Math.sin(2 * Math.PI * 0.5 * t); // 0.5 Hz oscillation
    const alpha = alpha_base + alpha_variation + vortexComponent * 5;
    
    // Generate instantaneous loads
    const loads = generateThinAirfoilLoads({
      velocity: V_kmh,
      angleOfAttack: alpha,
      chord,
      span,
      numPoints: 20, // Reduced for speed
    });
    
    timeSeries.push({
      time: t,
      velocity: V_kmh,
      alpha,
      lift: loads.forces.lift,
      drag: loads.forces.drag,
      cl: loads.coefficients.cl,
      cd: loads.coefficients.cd,
    });
  }
  
  return {
    config,
    timeSeries,
    statistics: {
      meanLift: timeSeries.reduce((sum, d) => sum + d.lift, 0) / numSteps,
      meanDrag: timeSeries.reduce((sum, d) => sum + d.drag, 0) / numSteps,
      maxLift: Math.max(...timeSeries.map(d => d.lift)),
      minLift: Math.min(...timeSeries.map(d => d.lift)),
    },
    metadata: {
      method: 'transient_loads',
      timestamp: new Date().toISOString(),
    },
  };
};

/**
 * Export loads to JSON format
 * @param {Object} loads - Load data object
 * @param {string} filename - Output filename
 */
export const exportToJSON = (loads, filename = 'aero_loads.json') => {
  const blob = new Blob([JSON.stringify(loads, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};

/**
 * Export loads to CSV format
 * @param {Object} loads - Load data object
 * @param {string} filename - Output filename
 */
export const exportToCSV = (loads, filename = 'aero_loads.csv') => {
  let csv = '';
  
  if (loads.distribution) {
    // Pressure distribution CSV
    csv = 'x,x_norm,cp_upper,cp_lower,p_upper,p_lower,yc,yt_upper,yt_lower\n';
    loads.distribution.forEach(point => {
      csv += `${point.x},${point.x_normalized},${point.cp_upper},${point.cp_lower},`;
      csv += `${point.p_upper},${point.p_lower},${point.yc},${point.yt_upper},${point.yt_lower}\n`;
    });
  } else if (loads.timeSeries) {
    // Time series CSV
    csv = 'time,velocity,alpha,lift,drag,cl,cd\n';
    loads.timeSeries.forEach(point => {
      csv += `${point.time},${point.velocity},${point.alpha},${point.lift},`;
      csv += `${point.drag},${point.cl},${point.cd}\n`;
    });
  } else if (loads.spanwiseDistribution) {
    // Spanwise distribution CSV
    csv = 'y,y_norm,cl_local,lift_local,circulation\n';
    loads.spanwiseDistribution.forEach(point => {
      csv += `${point.y},${point.y_normalized},${point.cl_local},`;
      csv += `${point.lift_local},${point.circulation}\n`;
    });
  }
  
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};

/**
 * Export loads to HDF5-compatible JSON (for Python consumption)
 * @param {Object} loads - Load data object
 * @param {string} filename - Output filename
 */
export const exportToHDF5JSON = (loads, filename = 'aero_loads_hdf5.json') => {
  const hdf5Structure = {
    metadata: loads.metadata,
    config: loads.config,
    aerodynamics: loads.aerodynamics,
    forces: loads.forces,
    coefficients: loads.coefficients,
    datasets: {},
  };
  
  if (loads.distribution) {
    hdf5Structure.datasets.pressure_distribution = {
      shape: [loads.distribution.length, 9],
      dtype: 'float64',
      data: loads.distribution.map(p => [
        p.x, p.x_normalized, p.cp_upper, p.cp_lower,
        p.p_upper, p.p_lower, p.yc, p.yt_upper, p.yt_lower
      ]),
      columns: ['x', 'x_norm', 'cp_upper', 'cp_lower', 'p_upper', 'p_lower', 'yc', 'yt_upper', 'yt_lower'],
    };
  }
  
  if (loads.timeSeries) {
    hdf5Structure.datasets.time_series = {
      shape: [loads.timeSeries.length, 7],
      dtype: 'float64',
      data: loads.timeSeries.map(p => [
        p.time, p.velocity, p.alpha, p.lift, p.drag, p.cl, p.cd
      ]),
      columns: ['time', 'velocity', 'alpha', 'lift', 'drag', 'cl', 'cd'],
    };
  }
  
  if (loads.spanwiseDistribution) {
    hdf5Structure.datasets.spanwise = {
      shape: [loads.spanwiseDistribution.length, 5],
      dtype: 'float64',
      data: loads.spanwiseDistribution.map(p => [
        p.y, p.y_normalized, p.cl_local, p.lift_local, p.circulation
      ]),
      columns: ['y', 'y_norm', 'cl_local', 'lift_local', 'circulation'],
    };
  }
  
  exportToJSON(hdf5Structure, filename);
};

export default {
  generateThinAirfoilLoads,
  generatePanelMethodLoads,
  generateVLMLoads,
  generateTransientLoads,
  exportToJSON,
  exportToCSV,
  exportToHDF5JSON,
};
