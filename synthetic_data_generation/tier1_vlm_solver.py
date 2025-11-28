"""
Tier 1: Fast VLM (Vortex Lattice Method) solver for bulk data generation.
Generates thousands of samples quickly with approximate aerodynamic outputs.
"""

import numpy as np
from typing import Tuple, Optional
import time
from dataclasses import dataclass

from schema import (
    AeroSample, GeometryParameters, FlowConditions, SimulationState,
    GlobalOutputs, FieldOutputs, ProvenanceMetadata, FidelityTier,
    SolverType, generate_sample_id
)
from tier0_geometry import F1GeometryGenerator, PanelMesh


class VLMSolver:
    """
    Vortex Lattice Method solver for lifting surfaces.
    Implements classical VLM with ground effect approximation.
    """
    
    def __init__(self, mesh: PanelMesh, flow: FlowConditions):
        """Initialize VLM solver with mesh and flow conditions."""
        self.mesh = mesh
        self.flow = flow
        self.n_panels = len(mesh.panels)
        
        # Vortex strengths (circulation)
        self.Gamma = np.zeros(self.n_panels)
        
        # Influence coefficient matrix
        self.AIC = None
        
    def compute_influence_coefficients(self):
        """
        Compute aerodynamic influence coefficient (AIC) matrix.
        AIC[i,j] = induced velocity at panel i due to unit vortex at panel j.
        """
        print(f"Computing AIC matrix ({self.n_panels}x{self.n_panels})...")
        start = time.time()
        
        self.AIC = np.zeros((self.n_panels, self.n_panels))
        
        for i in range(self.n_panels):
            # Control point (panel center)
            cp = self.mesh.panel_centers[i]
            normal = self.mesh.panel_normals[i]
            
            for j in range(self.n_panels):
                # Vortex panel j
                panel_verts = self.mesh.vertices[self.mesh.panels[j]]
                
                # Compute induced velocity from horseshoe vortex
                v_ind = self._horseshoe_vortex_velocity(cp, panel_verts)
                
                # Ground effect (mirror vortex)
                if self.flow.ground_gap < 0.1:  # Close to ground
                    cp_mirror = cp.copy()
                    cp_mirror[1] = -cp_mirror[1]  # Mirror in y
                    v_ind_mirror = self._horseshoe_vortex_velocity(cp_mirror, panel_verts)
                    v_ind -= v_ind_mirror  # Subtract mirror effect
                
                # Project onto normal (boundary condition: no flow through surface)
                self.AIC[i, j] = np.dot(v_ind, normal)
        
        elapsed = time.time() - start
        print(f"AIC computed in {elapsed:.2f}s")
        
    def _horseshoe_vortex_velocity(
        self,
        point: np.ndarray,
        panel_verts: np.ndarray
    ) -> np.ndarray:
        """
        Compute induced velocity at a point from a horseshoe vortex.
        Simplified implementation using Biot-Savart law.
        """
        # Use simplified horseshoe: bound vortex + 2 trailing vortices
        # For speed, use approximate formulation
        
        # Bound vortex (quarter-chord line)
        v0 = panel_verts[0]
        v1 = panel_verts[1]
        v2 = panel_verts[2]
        v3 = panel_verts[3]
        
        # Quarter chord points
        qc1 = 0.75 * v0 + 0.25 * v1
        qc2 = 0.75 * v3 + 0.25 * v2
        
        # Bound vortex contribution
        v_bound = self._vortex_segment_velocity(point, qc1, qc2)
        
        # Trailing vortices (extend to infinity, approximate as far downstream)
        far_downstream = 1000.0  # Large distance
        trail_dir = np.array([1.0, 0.0, 0.0])  # Streamwise direction
        
        trail1_end = qc1 + far_downstream * trail_dir
        trail2_end = qc2 + far_downstream * trail_dir
        
        v_trail1 = self._vortex_segment_velocity(point, qc1, trail1_end)
        v_trail2 = -self._vortex_segment_velocity(point, qc2, trail2_end)
        
        return v_bound + v_trail1 + v_trail2
    
    def _vortex_segment_velocity(
        self,
        point: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray
    ) -> np.ndarray:
        """
        Compute induced velocity from a straight vortex segment using Biot-Savart.
        """
        r1 = point - v1
        r2 = point - v2
        r1_mag = np.linalg.norm(r1) + 1e-10
        r2_mag = np.linalg.norm(r2) + 1e-10
        
        r1_cross_r2 = np.cross(r1, r2)
        r1_cross_r2_mag = np.linalg.norm(r1_cross_r2) + 1e-10
        
        # Biot-Savart formula
        cos_theta1 = np.dot(r1, v2 - v1) / (r1_mag * np.linalg.norm(v2 - v1) + 1e-10)
        cos_theta2 = np.dot(r2, v2 - v1) / (r2_mag * np.linalg.norm(v2 - v1) + 1e-10)
        
        K = (1.0 / (4.0 * np.pi)) * (cos_theta1 - cos_theta2) / r1_cross_r2_mag**2
        
        return K * r1_cross_r2
    
    def solve(self) -> Tuple[np.ndarray, GlobalOutputs, FieldOutputs]:
        """
        Solve VLM system for vortex strengths and compute forces.
        
        Returns:
            Gamma: Vortex strengths
            global_outputs: Global force coefficients
            field_outputs: Field data (Cp, etc.)
        """
        print("Solving VLM system...")
        start = time.time()
        
        # Build AIC if not already computed
        if self.AIC is None:
            self.compute_influence_coefficients()
        
        # Right-hand side: freestream velocity projected onto normals
        V_inf = np.array([self.flow.V_inf, 0.0, 0.0])  # Streamwise
        rhs = np.array([
            -np.dot(V_inf, normal) 
            for normal in self.mesh.panel_normals
        ])
        
        # Solve linear system: AIC * Gamma = rhs
        self.Gamma = np.linalg.solve(self.AIC, rhs)
        
        # Compute forces
        global_outputs, field_outputs = self._compute_forces()
        
        elapsed = time.time() - start
        print(f"VLM solved in {elapsed:.2f}s")
        
        return self.Gamma, global_outputs, field_outputs
    
    def _compute_forces(self) -> Tuple[GlobalOutputs, FieldOutputs]:
        """Compute aerodynamic forces from vortex strengths."""
        
        # Kutta-Joukowski theorem: F = rho * V x Gamma
        V_inf = np.array([self.flow.V_inf, 0.0, 0.0])
        
        # Force on each panel
        forces = np.zeros((self.n_panels, 3))
        Cp = np.zeros(self.n_panels)
        
        for i in range(self.n_panels):
            # Circulation vector (along span)
            panel_verts = self.mesh.vertices[self.mesh.panels[i]]
            span_vec = panel_verts[3] - panel_verts[0]
            span_vec = span_vec / (np.linalg.norm(span_vec) + 1e-10)
            
            Gamma_vec = self.Gamma[i] * span_vec
            
            # Force per unit span
            dF = self.flow.rho * np.cross(V_inf, Gamma_vec) * self.mesh.panel_areas[i]
            forces[i] = dF
            
            # Pressure coefficient (approximate)
            v_local = np.linalg.norm(V_inf + self.Gamma[i] * span_vec)
            Cp[i] = 1.0 - (v_local / self.flow.V_inf)**2
        
        # Sum forces
        total_force = forces.sum(axis=0)
        L = -total_force[1]  # Lift (negative y direction = downforce)
        D = total_force[0]  # Drag (x direction)
        
        # Reference area (approximate as sum of panel areas)
        S_ref = self.mesh.panel_areas.sum()
        q_inf = 0.5 * self.flow.rho * self.flow.V_inf**2
        
        # Coefficients
        CL = L / (q_inf * S_ref)
        CD_induced = D / (q_inf * S_ref)
        CD_total = CD_induced * 1.15  # Add approximate profile drag
        
        # Downforce split (front vs rear)
        front_mask = self.mesh.component_labels <= 4  # Front wing components
        rear_mask = self.mesh.component_labels >= 5  # Rear wing components
        
        downforce_front = -forces[front_mask, 1].sum()
        downforce_rear = -forces[rear_mask, 1].sum()
        balance = downforce_front / (downforce_front + downforce_rear + 1e-10)
        
        # L/D ratio
        L_over_D = abs(L / (D + 1e-10))
        
        global_outputs = GlobalOutputs(
            CL=CL,
            CD_total=CD_total,
            CD_induced=CD_induced,
            L=L,
            D=D,
            downforce_front=downforce_front,
            downforce_rear=downforce_rear,
            balance=balance,
            L_over_D=L_over_D
        )
        
        field_outputs = FieldOutputs(
            Cp=Cp,
            Gamma=self.Gamma,
            panel_centers=self.mesh.panel_centers,
            panel_normals=self.mesh.panel_normals,
            panel_areas=self.mesh.panel_areas
        )
        
        return global_outputs, field_outputs


def run_vlm_simulation(
    geom_params: GeometryParameters,
    flow_conditions: FlowConditions,
    sample_index: int = 0
) -> AeroSample:
    """
    Run a complete VLM simulation and return AeroSample.
    
    Args:
        geom_params: Geometry parameters
        flow_conditions: Flow conditions
        sample_index: Sample index for ID generation
    
    Returns:
        Complete AeroSample with results
    """
    start_time = time.time()
    
    # Generate geometry
    generator = F1GeometryGenerator(geom_params)
    mesh = generator.generate_complete_geometry()
    
    # Create VLM solver
    solver = VLMSolver(mesh, flow_conditions)
    
    # Solve
    Gamma, global_outputs, field_outputs = solver.solve()
    
    runtime = time.time() - start_time
    
    # Create sample
    sample = AeroSample(
        sample_id=generate_sample_id(FidelityTier.TIER_1_FAST_PHYSICS, sample_index),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        fidelity_tier=FidelityTier.TIER_1_FAST_PHYSICS,
        geometry_params=geom_params,
        flow_conditions=flow_conditions,
        state=SimulationState(is_steady=True, is_transient=False),
        global_outputs=global_outputs,
        field_outputs=field_outputs,
        provenance=ProvenanceMetadata(
            solver=SolverType.VLM,
            mesh_size=len(mesh.panels),
            runtime_seconds=runtime,
            convergence_achieved=True,
            notes=f"VLM with {len(mesh.panels)} panels"
        )
    )
    
    return sample


def test_vlm_solver():
    """Test VLM solver."""
    print("=== Testing Tier 1: VLM Solver ===\n")
    
    # Create test geometry
    geom = GeometryParameters(
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
    
    # Flow conditions
    flow = FlowConditions.standard_conditions(V_inf=70.0)
    
    # Run simulation
    sample = run_vlm_simulation(geom, flow, sample_index=0)
    
    print("\n=== Results ===")
    print(f"Sample ID: {sample.sample_id}")
    print(f"Runtime: {sample.provenance.runtime_seconds:.2f}s")
    print(f"Mesh size: {sample.provenance.mesh_size} panels")
    print(f"\nGlobal outputs:")
    print(f"  CL = {sample.global_outputs.CL:.3f}")
    print(f"  CD = {sample.global_outputs.CD_total:.3f}")
    print(f"  L/D = {sample.global_outputs.L_over_D:.2f}")
    print(f"  Downforce: {sample.global_outputs.L:.1f} N")
    print(f"  Balance: {sample.global_outputs.balance:.1%} front")
    
    # Save JSON
    with open("test_vlm_sample.json", "w") as f:
        f.write(sample.to_json())
    print(f"\nSaved to test_vlm_sample.json")
    
    return sample


if __name__ == "__main__":
    sample = test_vlm_solver()
