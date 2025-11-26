"""
Graph Builder for GNN-RANS
Convert unstructured meshes to PyTorch Geometric graphs
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import Dict, Any, Tuple, List
import trimesh


class MeshToGraphConverter:
    """
    Convert unstructured CFD meshes to graphs for GNN processing
    
    Supports:
    - Tetrahedral meshes
    - Hexahedral meshes
    - Mixed element types
    """
    
    def __init__(self):
        self.node_feature_names = ['x', 'y', 'z', 'boundary_type', 'volume', 'wall_dist']
        self.edge_feature_names = ['nx', 'ny', 'nz', 'area']
    
    def convert_mesh_to_graph(
        self,
        vertices: np.ndarray,
        cells: np.ndarray,
        boundary_info: Dict[str, Any] = None
    ) -> Data:
        """
        Convert mesh to PyTorch Geometric Data object
        
        Args:
            vertices: (N, 3) - Vertex coordinates
            cells: (C, 4) or (C, 8) - Cell connectivity (tet or hex)
            boundary_info: Optional boundary condition information
        
        Returns:
            PyTorch Geometric Data object
        """
        num_nodes = len(vertices)
        
        # Build node features
        node_features = self._build_node_features(vertices, cells, boundary_info)
        
        # Build edge connectivity and features
        edge_index, edge_features = self._build_edges(vertices, cells)
        
        # Create PyG Data object
        data = Data(
            x=torch.from_numpy(node_features).float(),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(edge_features).float(),
            num_nodes=num_nodes
        )
        
        return data
    
    def _build_node_features(
        self,
        vertices: np.ndarray,
        cells: np.ndarray,
        boundary_info: Dict[str, Any] = None
    ) -> np.ndarray:
        """
        Build node feature matrix
        
        Features:
        - x, y, z: Coordinates
        - boundary_type: 0=internal, 1=wall, 2=inlet, 3=outlet
        - volume: Cell volume (averaged for nodes)
        - wall_dist: Distance to nearest wall
        """
        num_nodes = len(vertices)
        node_features = np.zeros((num_nodes, 6))
        
        # Coordinates
        node_features[:, 0:3] = vertices
        
        # Boundary type (default: internal)
        node_features[:, 3] = 0
        
        if boundary_info:
            # Mark boundary nodes
            if 'wall_nodes' in boundary_info:
                node_features[boundary_info['wall_nodes'], 3] = 1
            if 'inlet_nodes' in boundary_info:
                node_features[boundary_info['inlet_nodes'], 3] = 2
            if 'outlet_nodes' in boundary_info:
                node_features[boundary_info['outlet_nodes'], 3] = 3
        
        # Cell volumes (compute and average to nodes)
        cell_volumes = self._compute_cell_volumes(vertices, cells)
        node_volumes = self._average_to_nodes(cell_volumes, cells, num_nodes)
        node_features[:, 4] = node_volumes
        
        # Wall distance (simplified - would use proper distance computation)
        wall_dist = self._compute_wall_distance(vertices, boundary_info)
        node_features[:, 5] = wall_dist
        
        return node_features
    
    def _build_edges(
        self,
        vertices: np.ndarray,
        cells: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build edge connectivity and features
        
        Returns:
            edge_index: (2, E) - Edge connectivity
            edge_features: (E, 4) - Edge features [nx, ny, nz, area]
        """
        edges = set()
        edge_features_list = []
        
        # For each cell, connect all pairs of vertices
        for cell in cells:
            n_vertices = len(cell)
            
            # Connect all vertex pairs in the cell
            for i in range(n_vertices):
                for j in range(i + 1, n_vertices):
                    v1, v2 = cell[i], cell[j]
                    
                    # Add bidirectional edges
                    if (v1, v2) not in edges:
                        edges.add((v1, v2))
                        edges.add((v2, v1))
                        
                        # Compute edge features (face normal and area)
                        edge_vec = vertices[v2] - vertices[v1]
                        edge_length = np.linalg.norm(edge_vec)
                        
                        if edge_length > 1e-10:
                            normal = edge_vec / edge_length
                        else:
                            normal = np.array([0, 0, 0])
                        
                        # Approximate face area (simplified)
                        area = edge_length ** 2
                        
                        # Store features for both directions
                        features = np.array([normal[0], normal[1], normal[2], area])
                        edge_features_list.append(features)
                        edge_features_list.append(features)  # Same for reverse edge
        
        # Convert to arrays
        edge_index = np.array(list(edges)).T
        edge_features = np.array(edge_features_list)
        
        return edge_index, edge_features
    
    def _compute_cell_volumes(
        self,
        vertices: np.ndarray,
        cells: np.ndarray
    ) -> np.ndarray:
        """Compute cell volumes"""
        volumes = np.zeros(len(cells))
        
        for i, cell in enumerate(cells):
            if len(cell) == 4:  # Tetrahedral
                # Compute tet volume
                v0, v1, v2, v3 = vertices[cell]
                volumes[i] = np.abs(np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6.0
            elif len(cell) == 8:  # Hexahedral
                # Approximate hex volume
                v0, v7 = vertices[cell[0]], vertices[cell[7]]
                volumes[i] = np.linalg.norm(v7 - v0) ** 3
            else:
                volumes[i] = 1.0  # Default
        
        return volumes
    
    def _average_to_nodes(
        self,
        cell_values: np.ndarray,
        cells: np.ndarray,
        num_nodes: int
    ) -> np.ndarray:
        """Average cell values to nodes"""
        node_values = np.zeros(num_nodes)
        node_counts = np.zeros(num_nodes)
        
        for cell_idx, cell in enumerate(cells):
            for node_idx in cell:
                node_values[node_idx] += cell_values[cell_idx]
                node_counts[node_idx] += 1
        
        # Avoid division by zero
        node_counts[node_counts == 0] = 1
        node_values /= node_counts
        
        return node_values
    
    def _compute_wall_distance(
        self,
        vertices: np.ndarray,
        boundary_info: Dict[str, Any] = None
    ) -> np.ndarray:
        """Compute distance to nearest wall (simplified)"""
        num_nodes = len(vertices)
        wall_dist = np.ones(num_nodes) * 1.0  # Default distance
        
        if boundary_info and 'wall_nodes' in boundary_info:
            wall_nodes = boundary_info['wall_nodes']
            wall_vertices = vertices[wall_nodes]
            
            # Compute distance to nearest wall node
            for i in range(num_nodes):
                if i in wall_nodes:
                    wall_dist[i] = 0.0
                else:
                    dists = np.linalg.norm(wall_vertices - vertices[i], axis=1)
                    wall_dist[i] = np.min(dists)
        
        return wall_dist


def create_mock_mesh(num_nodes: int = 1000) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Create mock unstructured mesh for testing
    
    Returns:
        vertices, cells, boundary_info
    """
    # Generate random vertices in a box
    vertices = np.random.rand(num_nodes, 3) * 2 - 1  # [-1, 1]^3
    
    # Generate tetrahedral cells (simplified)
    num_cells = num_nodes // 4
    cells = []
    for i in range(num_cells):
        # Random tetrahedron
        cell_nodes = np.random.choice(num_nodes, 4, replace=False)
        cells.append(cell_nodes)
    
    cells = np.array(cells)
    
    # Mark boundary nodes (nodes near surfaces)
    wall_nodes = np.where(np.abs(vertices[:, 2] + 1) < 0.1)[0]  # Bottom surface
    inlet_nodes = np.where(vertices[:, 0] < -0.9)[0]  # Left surface
    outlet_nodes = np.where(vertices[:, 0] > 0.9)[0]  # Right surface
    
    boundary_info = {
        'wall_nodes': wall_nodes,
        'inlet_nodes': inlet_nodes,
        'outlet_nodes': outlet_nodes
    }
    
    return vertices, cells, boundary_info


if __name__ == "__main__":
    # Test mesh to graph conversion
    converter = MeshToGraphConverter()
    
    # Create mock mesh
    vertices, cells, boundary_info = create_mock_mesh(num_nodes=1000)
    
    print(f"Mesh: {len(vertices)} nodes, {len(cells)} cells")
    
    # Convert to graph
    graph_data = converter.convert_mesh_to_graph(vertices, cells, boundary_info)
    
    print(f"\nGraph:")
    print(f"  Nodes: {graph_data.num_nodes}")
    print(f"  Edges: {graph_data.edge_index.shape[1]}")
    print(f"  Node features: {graph_data.x.shape}")
    print(f"  Edge features: {graph_data.edge_attr.shape}")
    
    # Check boundary nodes
    boundary_mask = graph_data.x[:, 3] > 0
    print(f"  Boundary nodes: {boundary_mask.sum().item()}")
