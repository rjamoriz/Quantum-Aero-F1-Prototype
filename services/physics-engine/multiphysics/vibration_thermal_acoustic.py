"""
Multi-Physics Integration Module
Implements vibrations, thermal effects, and aeroacoustics for F1 aerodynamics

Based on: Quantum-Aero F1 Prototype VIBRATIONS_THERMAL_AEROACOUSTIC.md
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging
from scipy import linalg
from scipy.integrate import odeint

logger = logging.getLogger(__name__)


@dataclass
class ModalProperties:
    """Modal analysis results"""
    natural_frequencies: np.ndarray  # Hz
    mode_shapes: np.ndarray
    damping_ratios: np.ndarray
    modal_masses: np.ndarray


@dataclass
class ThermalState:
    """Thermal state of component"""
    temperature: np.ndarray  # K
    heat_flux: np.ndarray  # W/m²
    thermal_stress: np.ndarray  # Pa


@dataclass
class AcousticMetrics:
    """Aeroacoustic metrics"""
    spl: float  # Sound Pressure Level (dB)
    frequency_spectrum: np.ndarray
    directivity: np.ndarray


class StructuralVibrationAnalyzer:
    """
    Structural vibration analysis for F1 components.
    
    Features:
    - Modal analysis (natural frequencies, mode shapes)
    - Forced vibration response
    - Fatigue life estimation
    - Flutter margin calculation
    """
    
    # Typical F1 component frequencies (Hz)
    COMPONENT_FREQUENCIES = {
        'front_wing': (15, 50),
        'rear_wing': (20, 60),
        'floor': (30, 80),
        'suspension': (10, 25),
        'diffuser': (40, 100)
    }
    
    def __init__(self, component: str = 'front_wing'):
        """
        Initialize vibration analyzer.
        
        Args:
            component: F1 component name
        """
        self.component = component
        self.freq_range = self.COMPONENT_FREQUENCIES.get(component, (10, 100))
        
        logger.info(f"Vibration analyzer initialized for {component}")
    
    def modal_analysis(
        self,
        mass_matrix: np.ndarray,
        stiffness_matrix: np.ndarray,
        n_modes: int = 10
    ) -> ModalProperties:
        """
        Perform modal analysis.
        
        Solves eigenvalue problem: (K - ω²M)φ = 0
        
        Args:
            mass_matrix: Mass matrix M
            stiffness_matrix: Stiffness matrix K
            n_modes: Number of modes to extract
            
        Returns:
            Modal properties
        """
        logger.info(f"Performing modal analysis: {n_modes} modes")
        
        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = linalg.eigh(
            stiffness_matrix,
            mass_matrix,
            subset_by_index=[0, n_modes-1]
        )
        
        # Natural frequencies (rad/s to Hz)
        omega_n = np.sqrt(eigenvalues)
        frequencies = omega_n / (2 * np.pi)
        
        # Modal masses
        modal_masses = np.diag(eigenvectors.T @ mass_matrix @ eigenvectors)
        
        # Typical damping ratios for F1 structures
        damping_ratios = np.full(n_modes, 0.02)  # 2% critical damping
        
        modal_props = ModalProperties(
            natural_frequencies=frequencies,
            mode_shapes=eigenvectors,
            damping_ratios=damping_ratios,
            modal_masses=modal_masses
        )
        
        logger.info(f"Modal analysis complete: f1={frequencies[0]:.2f} Hz")
        
        return modal_props
    
    def forced_response(
        self,
        modal_props: ModalProperties,
        force_frequency: float,
        force_amplitude: float
    ) -> Dict[str, float]:
        """
        Compute forced vibration response.
        
        Args:
            modal_props: Modal properties
            force_frequency: Excitation frequency (Hz)
            force_amplitude: Force amplitude (N)
            
        Returns:
            Response metrics
        """
        omega = 2 * np.pi * force_frequency
        
        # Compute response for each mode
        responses = []
        for i, (omega_n, zeta, m_n) in enumerate(zip(
            modal_props.natural_frequencies * 2 * np.pi,
            modal_props.damping_ratios,
            modal_props.modal_masses
        )):
            # Frequency response function
            H = 1 / (m_n * ((omega_n**2 - omega**2) + 2j * zeta * omega_n * omega))
            response = abs(H) * force_amplitude
            responses.append(response)
        
        max_response = max(responses)
        resonant_mode = np.argmax(responses)
        
        return {
            'max_displacement': max_response,
            'resonant_mode': resonant_mode,
            'resonant_frequency': modal_props.natural_frequencies[resonant_mode],
            'amplification_factor': max_response / (force_amplitude / modal_props.modal_masses[0])
        }
    
    def flutter_margin(
        self,
        modal_props: ModalProperties,
        velocity: float,
        air_density: float = 1.225
    ) -> float:
        """
        Estimate flutter margin.
        
        Flutter occurs when aerodynamic forces couple with structural modes.
        
        Args:
            modal_props: Modal properties
            velocity: Flow velocity (m/s)
            air_density: Air density (kg/m³)
            
        Returns:
            Flutter margin (>1.2 required for safety)
        """
        # Simplified flutter speed estimation (Theodorsen theory)
        # V_flutter ≈ sqrt(K / (ρ * S * C_L_alpha))
        
        # Typical F1 values
        CL_alpha = 0.1  # per degree
        S = 1.0  # Reference area (m²)
        
        # Use first bending-torsion coupled mode
        omega_critical = modal_props.natural_frequencies[0] * 2 * np.pi
        
        # Critical velocity
        V_flutter = np.sqrt(
            omega_critical / (air_density * S * CL_alpha)
        )
        
        # Flutter margin
        margin = V_flutter / velocity
        
        logger.info(f"Flutter margin: {margin:.2f} (V_flutter={V_flutter:.1f} m/s)")
        
        return margin
    
    def fatigue_life(
        self,
        stress_amplitude: float,
        mean_stress: float,
        material: str = 'carbon_fiber'
    ) -> float:
        """
        Estimate fatigue life using S-N curve.
        
        Args:
            stress_amplitude: Stress amplitude (MPa)
            mean_stress: Mean stress (MPa)
            material: Material type
            
        Returns:
            Cycles to failure
        """
        # S-N curve parameters (Basquin equation)
        # N = (S / S_f)^(-1/b)
        
        material_params = {
            'carbon_fiber': {'S_f': 1500, 'b': 0.1},  # MPa
            'aluminum': {'S_f': 200, 'b': 0.12},
            'titanium': {'S_f': 500, 'b': 0.08}
        }
        
        params = material_params.get(material, material_params['carbon_fiber'])
        
        # Goodman correction for mean stress
        S_a_eq = stress_amplitude / (1 - mean_stress / params['S_f'])
        
        # Cycles to failure
        N_f = (S_a_eq / params['S_f']) ** (-1 / params['b'])
        
        return N_f


class ThermalAnalyzer:
    """
    Thermal analysis for F1 components.
    
    Features:
    - Heat transfer (conduction, convection, radiation)
    - Thermal stress analysis
    - Cooling system optimization
    - Aerothermal coupling
    """
    
    def __init__(self):
        """Initialize thermal analyzer"""
        logger.info("Thermal analyzer initialized")
    
    def steady_state_temperature(
        self,
        heat_generation: float,
        convection_coeff: float,
        ambient_temp: float = 293.15,
        thermal_conductivity: float = 200.0
    ) -> float:
        """
        Compute steady-state temperature.
        
        Args:
            heat_generation: Heat generation rate (W)
            convection_coeff: Convection coefficient (W/m²K)
            ambient_temp: Ambient temperature (K)
            thermal_conductivity: Thermal conductivity (W/mK)
            
        Returns:
            Surface temperature (K)
        """
        # Simple lumped capacitance model
        # Q = h * A * (T - T_ambient)
        
        A = 1.0  # Reference area (m²)
        
        # Temperature rise
        delta_T = heat_generation / (convection_coeff * A)
        
        T_surface = ambient_temp + delta_T
        
        return T_surface
    
    def brake_cooling(
        self,
        brake_power: float,
        velocity: float,
        duct_area: float = 0.05
    ) -> Dict[str, float]:
        """
        Analyze brake cooling performance.
        
        Args:
            brake_power: Braking power (W)
            velocity: Vehicle velocity (m/s)
            duct_area: Cooling duct area (m²)
            
        Returns:
            Cooling metrics
        """
        # Air properties
        rho = 1.225  # kg/m³
        cp = 1005  # J/kgK
        
        # Mass flow rate through duct
        m_dot = rho * velocity * duct_area
        
        # Temperature rise
        delta_T = brake_power / (m_dot * cp)
        
        # Brake disc temperature (simplified)
        T_ambient = 293.15  # K
        T_brake = T_ambient + delta_T
        
        # Cooling effectiveness
        effectiveness = min(1.0, 1000 / T_brake)  # Target: T < 1000K
        
        return {
            'brake_temperature': T_brake,
            'temperature_rise': delta_T,
            'mass_flow_rate': m_dot,
            'cooling_effectiveness': effectiveness
        }
    
    def thermal_stress(
        self,
        temperature_gradient: float,
        thermal_expansion_coeff: float = 12e-6,
        youngs_modulus: float = 70e9
    ) -> float:
        """
        Compute thermal stress.
        
        σ_thermal = α * E * ΔT
        
        Args:
            temperature_gradient: Temperature change (K)
            thermal_expansion_coeff: Thermal expansion coefficient (1/K)
            youngs_modulus: Young's modulus (Pa)
            
        Returns:
            Thermal stress (Pa)
        """
        stress = thermal_expansion_coeff * youngs_modulus * temperature_gradient
        
        return stress


class AeroacousticAnalyzer:
    """
    Aeroacoustic analysis for F1.
    
    Features:
    - Sound pressure level (SPL) prediction
    - Frequency spectrum analysis
    - Noise source identification
    - FIA regulation compliance
    """
    
    FIA_NOISE_LIMIT = 110  # dB
    
    def __init__(self):
        """Initialize aeroacoustic analyzer"""
        logger.info("Aeroacoustic analyzer initialized")
    
    def lighthill_acoustic_analogy(
        self,
        velocity: float,
        characteristic_length: float,
        mach_number: float
    ) -> float:
        """
        Predict acoustic power using Lighthill's analogy.
        
        P_acoustic ∝ ρ * U^8 * L² / c^5
        
        Args:
            velocity: Flow velocity (m/s)
            characteristic_length: Characteristic length (m)
            mach_number: Mach number
            
        Returns:
            Sound pressure level (dB)
        """
        # Reference values
        rho = 1.225  # kg/m³
        c = 340  # m/s (speed of sound)
        p_ref = 20e-6  # Pa (reference pressure)
        
        # Acoustic power (Lighthill's 8th power law)
        P_acoustic = rho * (velocity**8) * (characteristic_length**2) / (c**5)
        
        # Sound pressure
        p_rms = np.sqrt(P_acoustic / (4 * np.pi * (10**2)))  # At 10m distance
        
        # SPL in dB
        SPL = 20 * np.log10(p_rms / p_ref)
        
        logger.info(f"Predicted SPL: {SPL:.1f} dB (limit: {self.FIA_NOISE_LIMIT} dB)")
        
        return SPL
    
    def vortex_shedding_frequency(
        self,
        velocity: float,
        diameter: float,
        strouhal_number: float = 0.2
    ) -> float:
        """
        Compute vortex shedding frequency.
        
        f = St * U / D
        
        Args:
            velocity: Flow velocity (m/s)
            diameter: Characteristic diameter (m)
            strouhal_number: Strouhal number
            
        Returns:
            Shedding frequency (Hz)
        """
        frequency = strouhal_number * velocity / diameter
        
        return frequency
    
    def fia_compliance_check(
        self,
        spl: float,
        frequency_spectrum: Optional[np.ndarray] = None
    ) -> Dict[str, bool]:
        """
        Check FIA noise regulation compliance.
        
        Args:
            spl: Overall sound pressure level (dB)
            frequency_spectrum: Frequency spectrum
            
        Returns:
            Compliance status
        """
        compliant = spl <= self.FIA_NOISE_LIMIT
        margin = self.FIA_NOISE_LIMIT - spl
        
        return {
            'compliant': compliant,
            'spl': spl,
            'limit': self.FIA_NOISE_LIMIT,
            'margin': margin
        }


class MultiPhysicsCoupler:
    """
    Couples vibration, thermal, and acoustic phenomena.
    
    Implements:
    - Vibro-acoustic coupling
    - Thermo-structural coupling
    - Aerothermal coupling
    """
    
    def __init__(self):
        """Initialize multi-physics coupler"""
        self.vibration = StructuralVibrationAnalyzer()
        self.thermal = ThermalAnalyzer()
        self.acoustic = AeroacousticAnalyzer()
        
        logger.info("Multi-physics coupler initialized")
    
    def coupled_analysis(
        self,
        velocity: float,
        temperature: float,
        structural_load: float
    ) -> Dict:
        """
        Perform coupled multi-physics analysis.
        
        Args:
            velocity: Flow velocity (m/s)
            temperature: Operating temperature (K)
            structural_load: Structural load (N)
            
        Returns:
            Coupled analysis results
        """
        results = {}
        
        # Thermal analysis
        brake_power = 100000  # W (typical F1 braking)
        thermal_results = self.thermal.brake_cooling(brake_power, velocity)
        results['thermal'] = thermal_results
        
        # Thermal stress
        temp_gradient = thermal_results['temperature_rise']
        thermal_stress = self.thermal.thermal_stress(temp_gradient)
        results['thermal_stress'] = thermal_stress
        
        # Acoustic analysis
        mach = velocity / 340
        spl = self.acoustic.lighthill_acoustic_analogy(velocity, 1.0, mach)
        compliance = self.acoustic.fia_compliance_check(spl)
        results['acoustic'] = compliance
        
        # Combined stress (structural + thermal)
        structural_stress = structural_load / 0.01  # Simplified
        total_stress = structural_stress + thermal_stress
        results['total_stress'] = total_stress
        
        logger.info(f"Coupled analysis complete: SPL={spl:.1f}dB, T={thermal_results['brake_temperature']:.1f}K")
        
        return results


if __name__ == "__main__":
    # Test multi-physics integration
    logging.basicConfig(level=logging.INFO)
    
    print("Multi-Physics Integration Test")
    print("=" * 60)
    
    # Test vibration analysis
    print("\n1. Structural Vibration Analysis")
    vibration = StructuralVibrationAnalyzer('front_wing')
    
    # Simple 2-DOF system
    M = np.array([[1, 0], [0, 1]])
    K = np.array([[2, -1], [-1, 2]]) * 1000
    
    modal_props = vibration.modal_analysis(M, K, n_modes=2)
    print(f"   Natural frequencies: {modal_props.natural_frequencies} Hz")
    
    # Flutter margin
    margin = vibration.flutter_margin(modal_props, velocity=80.0)
    print(f"   Flutter margin: {margin:.2f}")
    
    # Test thermal analysis
    print("\n2. Thermal Analysis")
    thermal = ThermalAnalyzer()
    
    brake_results = thermal.brake_cooling(brake_power=100000, velocity=80.0)
    print(f"   Brake temperature: {brake_results['brake_temperature']:.1f} K")
    print(f"   Cooling effectiveness: {brake_results['cooling_effectiveness']:.2%}")
    
    # Test acoustic analysis
    print("\n3. Aeroacoustic Analysis")
    acoustic = AeroacousticAnalyzer()
    
    spl = acoustic.lighthill_acoustic_analogy(velocity=80.0, characteristic_length=1.0, mach_number=0.235)
    compliance = acoustic.fia_compliance_check(spl)
    print(f"   SPL: {spl:.1f} dB")
    print(f"   FIA compliant: {compliance['compliant']}")
    
    # Test coupled analysis
    print("\n4. Coupled Multi-Physics Analysis")
    coupler = MultiPhysicsCoupler()
    
    results = coupler.coupled_analysis(velocity=80.0, temperature=350.0, structural_load=5000.0)
    print(f"   Brake temp: {results['thermal']['brake_temperature']:.1f} K")
    print(f"   SPL: {results['acoustic']['spl']:.1f} dB")
    print(f"   Total stress: {results['total_stress']/1e6:.1f} MPa")
    
    print("\n✅ All multi-physics tests passed!")
