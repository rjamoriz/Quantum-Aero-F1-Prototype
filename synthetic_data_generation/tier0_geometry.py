"""
Tier 0: Parametric geometry generation for F1-like configurations.
Creates mesh representations from parametric descriptions.
"""

import numpy as np
import trimesh
from typing import Tuple, List, Optional
from dataclasses import dataclass
from schema import GeometryParameters
import json


@dataclass
class PanelMesh:
    """Panel mesh representation for VLM/panel methods."""
    vertices: np.ndarray  # (n_vertices, 3)
    panels: np.ndarray  # (n_panels, 4) indices into vertices (quad panels)
    panel_centers: np.ndarray  # (n_panels, 3)
    panel_normals: np.ndarray  # (n_panels, 3)
    panel_areas: np.ndarray  # (n_panels,)
    component_labels: np.ndarray  # (n_panels,) component ID
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'n_vertices': len(self.vertices),
            'n_panels': len(self.panels),
            'vertices': self.vertices.tolist(),
            'panels': self.panels.tolist(),
            'panel_centers': self.panel_centers.tolist(),
            'panel_normals': self.panel_normals.tolist(),
            'panel_areas': self.panel_areas.tolist(),
            'component_labels': self.component_labels.tolist(),
        }
    
    def save_vtk(self, filename: str):
        """Save as VTK file for Paraview visualization."""
        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("F1 Panel Mesh\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")
            f.write(f"POINTS {len(self.vertices)} float\n")
            for v in self.vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
            
            f.write(f"\nPOLYGONS {len(self.panels)} {len(self.panels) * 5}\n")
            for panel in self.panels:
                f.write(f"4 {panel[0]} {panel[1]} {panel[2]} {panel[3]}\n")
            
            f.write(f"\nCELL_DATA {len(self.panels)}\n")
            f.write("SCALARS component_id int 1\n")
            f.write("LOOKUP_TABLE default\n")
            for label in self.component_labels:
                f.write(f"{label}\n")


class F1GeometryGenerator:
    """Generate parametric F1 aerodynamic component geometries."""
    
    COMPONENT_IDS = {
        'front_wing_main': 1,
        'front_wing_flap1': 2,
        'front_wing_flap2': 3,
        'front_wing_endplate': 4,
        'rear_wing_main': 5,
        'rear_wing_flap': 6,
        'beam_wing': 7,
        'floor': 8,
        'diffuser': 9,
        'sidepod': 10,
    }
    
    def __init__(self, params: GeometryParameters):
        """Initialize with geometry parameters."""
        self.params = params
    
    def generate_wing_section(
        self,
        chord: float,
        span: float,
        angle_deg: float,
        position: np.ndarray,
        n_chord: int = 20,
        n_span: int = 30,
        airfoil_thickness: float = 0.12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a wing section with NACA-like airfoil.
        
        Returns:
            vertices: (n_vertices, 3)
            panels: (n_panels, 4) quad panel indices
        """
        # Create airfoil profile (simplified NACA)
        x_airfoil = np.linspace(0, 1, n_chord)
        y_upper = airfoil_thickness * (0.2969 * np.sqrt(x_airfoil) 
                                        - 0.1260 * x_airfoil 
                                        - 0.3516 * x_airfoil**2 
                                        + 0.2843 * x_airfoil**3 
                                        - 0.1015 * x_airfoil**4)
        y_lower = -y_upper
        
        # Scale by chord
        x_airfoil *= chord / 100.0  # Convert cm to m
        y_upper *= chord / 100.0
        y_lower *= chord / 100.0
        
        # Create spanwise distribution
        z_span = np.linspace(-span/200.0, span/200.0, n_span)  # Convert cm to m, centered
        
        # Generate 3D mesh
        vertices = []
        for z in z_span:
            for i in range(n_chord):
                # Upper surface
                vertices.append([x_airfoil[i], y_upper[i], z])
            for i in range(n_chord-1, -1, -1):
                # Lower surface
                vertices.append([x_airfoil[i], y_lower[i], z])
        
        vertices = np.array(vertices)
        
        # Apply angle of attack rotation
        angle_rad = np.deg2rad(angle_deg)
        rot_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        vertices = vertices @ rot_matrix.T
        
        # Translate to position
        vertices += position
        
        # Create quad panels
        panels = []
        n_section = 2 * n_chord - 1  # Points per spanwise section
        for i_span in range(n_span - 1):
            for i_chord in range(n_section - 1):
                idx0 = i_span * n_section + i_chord
                idx1 = idx0 + 1
                idx2 = (i_span + 1) * n_section + i_chord + 1
                idx3 = (i_span + 1) * n_section + i_chord
                panels.append([idx0, idx1, idx2, idx3])
        
        panels = np.array(panels)
        
        return vertices, panels
    
    def generate_flat_plate(
        self,
        length: float,
        width: float,
        position: np.ndarray,
        angle_deg: float = 0.0,
        n_length: int = 20,
        n_width: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a flat plate (for floor, endplates, etc.)."""
        x = np.linspace(0, length/100.0, n_length)  # Convert cm to m
        z = np.linspace(-width/200.0, width/200.0, n_width)
        
        X, Z = np.meshgrid(x, z)
        Y = np.zeros_like(X)
        
        vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # Apply angle
        if angle_deg != 0:
            angle_rad = np.deg2rad(angle_deg)
            rot_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
            vertices = vertices @ rot_matrix.T
        
        # Translate
        vertices += position
        
        # Create panels
        panels = []
        for i in range(n_width - 1):
            for j in range(n_length - 1):
                idx0 = i * n_length + j
                idx1 = idx0 + 1
                idx2 = (i + 1) * n_length + j + 1
                idx3 = (i + 1) * n_length + j
                panels.append([idx0, idx1, idx2, idx3])
        
        panels = np.array(panels)
        
        return vertices, panels
    
    def compute_panel_properties(
        self,
        vertices: np.ndarray,
        panels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute panel centers, normals, and areas.
        
        Returns:
            centers: (n_panels, 3)
            normals: (n_panels, 3)
            areas: (n_panels,)
        """
        centers = []
        normals = []
        areas = []
        
        for panel in panels:
            # Get panel vertices
            v0, v1, v2, v3 = vertices[panel]
            
            # Center (average of 4 corners)
            center = (v0 + v1 + v2 + v3) / 4.0
            centers.append(center)
            
            # Normal (cross product of diagonals)
            d1 = v2 - v0
            d2 = v3 - v1
            normal = np.cross(d1, d2)
            area = np.linalg.norm(normal) / 2.0
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            
            normals.append(normal)
            areas.append(area)
        
        return np.array(centers), np.array(normals), np.array(areas)
    
    def generate_complete_geometry(self) -> PanelMesh:
        """Generate complete F1 geometry from parameters."""
        all_vertices = []
        all_panels = []
        all_labels = []
        vertex_offset = 0
        
        # 1. Front wing main plane
        v, p = self.generate_wing_section(
            chord=self.params.main_plane_chord,
            span=self.params.main_plane_span,
            angle_deg=self.params.main_plane_angle_deg,
            position=np.array([0.5, -0.05, 0.0]),  # Front of car
            n_chord=15,
            n_span=25
        )
        all_vertices.append(v)
        all_panels.append(p + vertex_offset)
        all_labels.extend([self.COMPONENT_IDS['front_wing_main']] * len(p))
        vertex_offset += len(v)
        
        # 2. Front wing flap 1
        v, p = self.generate_wing_section(
            chord=self.params.main_plane_chord * 0.6,
            span=self.params.main_plane_span * 0.9,
            angle_deg=self.params.flap1_angle_deg,
            position=np.array([0.5 + self.params.main_plane_chord/100.0, -0.08, 0.0]),
            n_chord=12,
            n_span=20
        )
        all_vertices.append(v)
        all_panels.append(p + vertex_offset)
        all_labels.extend([self.COMPONENT_IDS['front_wing_flap1']] * len(p))
        vertex_offset += len(v)
        
        # 3. Front wing flap 2
        v, p = self.generate_wing_section(
            chord=self.params.main_plane_chord * 0.4,
            span=self.params.main_plane_span * 0.8,
            angle_deg=self.params.flap2_angle_deg,
            position=np.array([0.5 + self.params.main_plane_chord/100.0 * 1.5, -0.11, 0.0]),
            n_chord=10,
            n_span=18
        )
        all_vertices.append(v)
        all_panels.append(p + vertex_offset)
        all_labels.extend([self.COMPONENT_IDS['front_wing_flap2']] * len(p))
        vertex_offset += len(v)
        
        # 4. Rear wing
        rear_x = 4.5  # Rear of car
        v, p = self.generate_wing_section(
            chord=self.params.rear_wing_chord,
            span=self.params.rear_wing_span,
            angle_deg=self.params.rear_wing_angle_deg,
            position=np.array([rear_x, 0.8, 0.0]),  # Elevated
            n_chord=15,
            n_span=20
        )
        all_vertices.append(v)
        all_panels.append(p + vertex_offset)
        all_labels.extend([self.COMPONENT_IDS['rear_wing_main']] * len(p))
        vertex_offset += len(v)
        
        # 5. Beam wing (if not DRS open)
        if not self.params.DRS_open:
            v, p = self.generate_wing_section(
                chord=self.params.rear_wing_chord * 0.5,
                span=self.params.rear_wing_span * 0.8,
                angle_deg=self.params.beam_wing_angle,
                position=np.array([rear_x - 0.2, 0.5, 0.0]),
                n_chord=10,
                n_span=15
            )
            all_vertices.append(v)
            all_panels.append(p + vertex_offset)
            all_labels.extend([self.COMPONENT_IDS['beam_wing']] * len(p))
            vertex_offset += len(v)
        
        # 6. Floor
        v, p = self.generate_flat_plate(
            length=400.0,  # 4m floor
            width=180.0,
            position=np.array([1.0, -self.params.floor_gap/1000.0, 0.0]),
            n_length=30,
            n_width=15
        )
        all_vertices.append(v)
        all_panels.append(p + vertex_offset)
        all_labels.extend([self.COMPONENT_IDS['floor']] * len(p))
        vertex_offset += len(v)
        
        # 7. Diffuser
        v, p = self.generate_flat_plate(
            length=self.params.diffuser_length,
            width=160.0,
            position=np.array([3.5, -self.params.floor_gap/1000.0, 0.0]),
            angle_deg=self.params.diffuser_angle,
            n_length=20,
            n_width=12
        )
        all_vertices.append(v)
        all_panels.append(p + vertex_offset)
        all_labels.extend([self.COMPONENT_IDS['diffuser']] * len(p))
        vertex_offset += len(v)
        
        # Combine all components
        vertices = np.vstack(all_vertices)
        panels = np.vstack(all_panels)
        labels = np.array(all_labels)
        
        # Compute panel properties
        centers, normals, areas = self.compute_panel_properties(vertices, panels)
        
        return PanelMesh(
            vertices=vertices,
            panels=panels,
            panel_centers=centers,
            panel_normals=normals,
            panel_areas=areas,
            component_labels=labels
        )


def test_geometry_generation():
    """Test geometry generation."""
    from schema import GeometryParameters
    
    print("=== Testing Tier 0: Geometry Generation ===\n")
    
    # Create test parameters
    params = GeometryParameters(
        main_plane_chord=100.0,
        main_plane_span=160.0,
        main_plane_angle_deg=5.0,
        flap1_angle_deg=10.0,
        flap2_angle_deg=15.0,
        endplate_height=300.0,
        rear_wing_chord=80.0,
        rear_wing_span=100.0,
        rear_wing_angle_deg=15.0,
        beam_wing_angle=8.0,
        floor_gap=25.0,
        diffuser_angle=18.0,
        diffuser_length=120.0,
        sidepod_width=45.0,
        sidepod_undercut=12.0,
        DRS_open=False
    )
    
    # Generate geometry
    generator = F1GeometryGenerator(params)
    mesh = generator.generate_complete_geometry()
    
    print(f"Generated mesh:")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Panels: {len(mesh.panels)}")
    print(f"  Total area: {mesh.panel_areas.sum():.3f} m²")
    print(f"  Components: {np.unique(mesh.component_labels)}")
    
    # Save VTK for visualization
    output_file = "test_f1_geometry.vtk"
    mesh.save_vtk(output_file)
    print(f"\nSaved to {output_file} (open in Paraview)")
    
    return mesh


if __name__ == "__main__":
    mesh = test_geometry_generation()
