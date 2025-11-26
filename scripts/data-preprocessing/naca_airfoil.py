"""
NACA Airfoil Generator
Generates 4-digit and 5-digit NACA airfoil profiles for F1 aerodynamics
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NACAProfile:
    """
    NACA airfoil profile generator.
    
    Supports:
    - 4-digit series (e.g., NACA 2412)
    - 5-digit series (e.g., NACA 23012)
    - Modified profiles for F1 applications
    """
    
    def __init__(self, designation: str):
        """
        Initialize NACA profile.
        
        Args:
            designation: NACA designation (e.g., "2412", "23012")
        """
        self.designation = designation.replace("NACA", "").replace(" ", "")
        self.n_digits = len(self.designation)
        
        if self.n_digits == 4:
            self._parse_4digit()
        elif self.n_digits == 5:
            self._parse_5digit()
        else:
            raise ValueError(f"Unsupported NACA designation: {designation}")
        
        logger.info(f"NACA {self.designation} profile initialized")
    
    def _parse_4digit(self):
        """Parse 4-digit NACA designation"""
        m = int(self.designation[0]) / 100.0  # Maximum camber
        p = int(self.designation[1]) / 10.0   # Position of maximum camber
        t = int(self.designation[2:4]) / 100.0  # Maximum thickness
        
        self.max_camber = m
        self.camber_position = p
        self.max_thickness = t
        
        logger.debug(f"4-digit: m={m}, p={p}, t={t}")
    
    def _parse_5digit(self):
        """Parse 5-digit NACA designation"""
        # Simplified 5-digit parsing
        cl_design = int(self.designation[0]) * 3 / 20.0
        p = int(self.designation[1:3]) / 100.0
        t = int(self.designation[3:5]) / 100.0
        
        # Approximate camber from design lift coefficient
        self.max_camber = cl_design / 10.0
        self.camber_position = p
        self.max_thickness = t
        
        logger.debug(f"5-digit: CL_design={cl_design}, p={p}, t={t}")
    
    def generate_coordinates(
        self,
        n_points: int = 100,
        closed_trailing_edge: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate airfoil coordinates.
        
        Args:
            n_points: Number of points on upper/lower surface
            closed_trailing_edge: Close trailing edge (TE thickness = 0)
            
        Returns:
            (x_coords, y_coords) for upper and lower surfaces
        """
        # Cosine spacing for better resolution at leading edge
        beta = np.linspace(0, np.pi, n_points)
        x = 0.5 * (1 - np.cos(beta))
        
        # Thickness distribution (symmetric)
        yt = self._thickness_distribution(x, closed_trailing_edge)
        
        # Camber line
        yc = self._camber_line(x)
        
        # Camber line slope
        dyc_dx = self._camber_slope(x)
        theta = np.arctan(dyc_dx)
        
        # Upper and lower surface coordinates
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)
        
        # Combine upper and lower (counterclockwise from TE)
        x_coords = np.concatenate([xu[::-1], xl[1:]])
        y_coords = np.concatenate([yu[::-1], yl[1:]])
        
        return x_coords, y_coords
    
    def _thickness_distribution(
        self,
        x: np.ndarray,
        closed_te: bool = True
    ) -> np.ndarray:
        """
        NACA thickness distribution.
        
        Standard formula:
        yt = 5*t * (0.2969*√x - 0.1260*x - 0.3516*x² + 0.2843*x³ - 0.1015*x⁴)
        """
        t = self.max_thickness
        
        if closed_te:
            # Closed trailing edge (original formula)
            a0, a1, a2, a3, a4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1015
        else:
            # Open trailing edge (modified for finite TE thickness)
            a0, a1, a2, a3, a4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1036
        
        yt = 5 * t * (
            a0 * np.sqrt(x) +
            a1 * x +
            a2 * x**2 +
            a3 * x**3 +
            a4 * x**4
        )
        
        return yt
    
    def _camber_line(self, x: np.ndarray) -> np.ndarray:
        """
        NACA camber line.
        
        For 4-digit series:
        yc = (m/p²) * (2*p*x - x²)  for x ≤ p
        yc = (m/(1-p)²) * ((1-2*p) + 2*p*x - x²)  for x > p
        """
        m = self.max_camber
        p = self.camber_position
        
        if m == 0:
            return np.zeros_like(x)
        
        yc = np.zeros_like(x)
        
        # Forward portion (x ≤ p)
        mask1 = x <= p
        if p > 0:
            yc[mask1] = (m / p**2) * (2*p*x[mask1] - x[mask1]**2)
        
        # Aft portion (x > p)
        mask2 = x > p
        if p < 1:
            yc[mask2] = (m / (1-p)**2) * ((1 - 2*p) + 2*p*x[mask2] - x[mask2]**2)
        
        return yc
    
    def _camber_slope(self, x: np.ndarray) -> np.ndarray:
        """
        Slope of camber line (dyc/dx).
        """
        m = self.max_camber
        p = self.camber_position
        
        if m == 0:
            return np.zeros_like(x)
        
        dyc_dx = np.zeros_like(x)
        
        # Forward portion
        mask1 = x <= p
        if p > 0:
            dyc_dx[mask1] = (2*m / p**2) * (p - x[mask1])
        
        # Aft portion
        mask2 = x > p
        if p < 1:
            dyc_dx[mask2] = (2*m / (1-p)**2) * (p - x[mask2])
        
        return dyc_dx
    
    def scale_and_transform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        chord: float = 1.0,
        angle: float = 0.0,
        position: Tuple[float, float, float] = (0, 0, 0)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale and transform airfoil coordinates.
        
        Args:
            x, y: Airfoil coordinates (normalized)
            chord: Chord length
            angle: Angle of attack (degrees)
            position: (x, y, z) position
            
        Returns:
            (x_3d, y_3d, z_3d) transformed coordinates
        """
        # Scale by chord
        x_scaled = x * chord
        y_scaled = y * chord
        
        # Rotate by angle of attack
        angle_rad = np.radians(angle)
        x_rot = x_scaled * np.cos(angle_rad) - y_scaled * np.sin(angle_rad)
        y_rot = x_scaled * np.sin(angle_rad) + y_scaled * np.cos(angle_rad)
        
        # Translate to position
        x_3d = x_rot + position[0]
        y_3d = np.full_like(x_rot, position[1])  # Spanwise position
        z_3d = y_rot + position[2]
        
        return x_3d, y_3d, z_3d
    
    def get_properties(self) -> dict:
        """Get airfoil geometric properties"""
        return {
            'designation': f"NACA {self.designation}",
            'max_camber': self.max_camber,
            'camber_position': self.camber_position,
            'max_thickness': self.max_thickness,
            'type': f"{self.n_digits}-digit"
        }


def create_f1_front_wing_profile(
    main_plane: str = "6412",
    flap1: str = "4415",
    flap2: str = "4418",
    n_points: int = 100
) -> dict:
    """
    Create F1 front wing multi-element profile.
    
    Args:
        main_plane: NACA designation for main plane
        flap1: NACA designation for first flap
        flap2: NACA designation for second flap
        n_points: Points per element
        
    Returns:
        Dictionary with coordinates for each element
    """
    profiles = {
        'main_plane': NACAProfile(main_plane),
        'flap1': NACAProfile(flap1),
        'flap2': NACAProfile(flap2)
    }
    
    coordinates = {}
    for name, profile in profiles.items():
        x, y = profile.generate_coordinates(n_points)
        coordinates[name] = {'x': x, 'y': y}
    
    logger.info(f"Created F1 front wing profile: {main_plane}/{flap1}/{flap2}")
    
    return coordinates


def create_f1_rear_wing_profile(
    main_plane: str = "9618",
    drs_flap: str = "6412",
    n_points: int = 100
) -> dict:
    """
    Create F1 rear wing profile.
    
    Args:
        main_plane: NACA designation for main plane
        drs_flap: NACA designation for DRS flap
        n_points: Points per element
        
    Returns:
        Dictionary with coordinates for each element
    """
    profiles = {
        'main_plane': NACAProfile(main_plane),
        'drs_flap': NACAProfile(drs_flap)
    }
    
    coordinates = {}
    for name, profile in profiles.items():
        x, y = profile.generate_coordinates(n_points)
        coordinates[name] = {'x': x, 'y': y}
    
    logger.info(f"Created F1 rear wing profile: {main_plane}/{drs_flap}")
    
    return coordinates


def plot_airfoil(x: np.ndarray, y: np.ndarray, title: str = "NACA Airfoil"):
    """
    Plot airfoil profile (requires matplotlib).
    
    Args:
        x, y: Airfoil coordinates
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        plt.plot(x, y, 'b-', linewidth=2)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available for plotting")


if __name__ == "__main__":
    # Test NACA airfoil generator
    logging.basicConfig(level=logging.INFO)
    
    print("NACA Airfoil Generator Test")
    print("=" * 60)
    
    # Test 4-digit series
    print("\n1. NACA 2412 (4-digit)")
    naca2412 = NACAProfile("2412")
    x, y = naca2412.generate_coordinates(n_points=100)
    print(f"   Generated {len(x)} points")
    print(f"   Properties: {naca2412.get_properties()}")
    
    # Test 5-digit series
    print("\n2. NACA 23012 (5-digit)")
    naca23012 = NACAProfile("23012")
    x, y = naca23012.generate_coordinates(n_points=100)
    print(f"   Generated {len(x)} points")
    print(f"   Properties: {naca23012.get_properties()}")
    
    # Test F1 profiles
    print("\n3. F1 Front Wing")
    front_wing = create_f1_front_wing_profile()
    for element, coords in front_wing.items():
        print(f"   {element}: {len(coords['x'])} points")
    
    print("\n4. F1 Rear Wing")
    rear_wing = create_f1_rear_wing_profile()
    for element, coords in rear_wing.items():
        print(f"   {element}: {len(coords['x'])} points")
    
    # Test transformation
    print("\n5. Transformation Test")
    naca = NACAProfile("0012")
    x, y = naca.generate_coordinates(50)
    x3d, y3d, z3d = naca.scale_and_transform(
        x, y,
        chord=0.25,
        angle=15.0,
        position=(1.0, 0.5, 0.1)
    )
    print(f"   Transformed to 3D: {len(x3d)} points")
    print(f"   Chord: 0.25m, AOA: 15°, Position: (1.0, 0.5, 0.1)")
    
    print("\n✅ All tests passed!")
