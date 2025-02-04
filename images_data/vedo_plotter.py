import vedo
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


vedo.settings.default_backend = 'vtk'

#load data

directory_name = 'images_data/2025-02-04_15:10:28_3_passings_10k'
gt_displacement = np.load(directory_name + '/ground_truth_displacement.npy')
pred_displacement = np.load(directory_name + '/prediction_displacement.npy')
gt_grid = np.load(directory_name + '/ground_truth_grid.npy')
pred_grid = np.load(directory_name + '/prediction_grid.npy')
gt_rest = np.load(directory_name + '/ground_truth_rest.npy')
pred_rest = np.load(directory_name + '/prediction_rest.npy')
gt_grid_rest = np.load(directory_name + '/ground_truth_grid_rest.npy')
pred_grid_rest = np.load(directory_name + '/prediction_grid_rest.npy')

#print_all_shapes
print('gt_displacement shape:', gt_displacement.shape)
print('pred_displacement shape:', pred_displacement.shape)
print('gt_grid shape:', gt_grid.shape)
print('pred_grid shape:', pred_grid.shape)
print('gt_rest shape:', gt_rest.shape)
print('pred_rest shape:', pred_rest.shape)
print('gt_grid_rest shape:', gt_grid_rest.shape)
print('pred_grid_rest shape:', pred_grid_rest.shape)



error = np.abs(gt_grid - pred_grid)
print('max error:', np.max(error))
print('mean error:', np.mean(error))
print('std error:', np.std(error))
print('error shape:', error.shape)


def interpolate_vector_field(old_grid, vector_field, new_grid):
        from scipy.interpolate import RBFInterpolator

        
        # Create RBF interpolators for each component

        # Remove duplicate points that could cause singularity
        unique_indices = np.unique(old_grid, axis=0, return_index=True)[1]
        old_grid_clean = old_grid[unique_indices]
        vector_field_clean = vector_field[unique_indices]
        try:
            rbf_u = RBFInterpolator(old_grid_clean, vector_field_clean[:, 0], 
                                kernel='cubic',
                                smoothing=1e-3)
            rbf_v = RBFInterpolator(old_grid_clean, vector_field_clean[:, 1],
                                kernel='cubic',
                                smoothing=1e-3)
            rbf_w = RBFInterpolator(old_grid_clean, vector_field_clean[:, 2],
                                kernel='cubic',
                                smoothing=1e-3)
            
            u_new = rbf_u(new_grid)
            v_new = rbf_v(new_grid)
            w_new = rbf_w(new_grid)
            
            return np.array([u_new, v_new, w_new]).T
            
        except np.linalg.LinAlgError:
            # Fallback to simpler kernel if still singular
            rbf_u = RBFInterpolator(old_grid_clean, vector_field_clean[:, 0], 
                                kernel='linear',
                                smoothing=1e-2)
            rbf_v = RBFInterpolator(old_grid_clean, vector_field_clean[:, 1],
                                kernel='linear',
                                smoothing=1e-2)
            rbf_w = RBFInterpolator(old_grid_clean, vector_field_clean[:, 2],
                                kernel='linear',
                                smoothing=1e-2)
            
            u_new = rbf_u(new_grid)
            v_new = rbf_v(new_grid)
            w_new = rbf_w(new_grid)
            
            return np.array([u_new, v_new, w_new]).T
        

interpolated_error = interpolate_vector_field(gt_grid_rest, error, pred_rest)

error_gt = np.linalg.norm(interpolated_error, axis=1)


import vedo
import numpy as np
import os
import vtk
import gmsh

def load_mesh_safe(mesh_path):
    """Load GMSH mesh and convert to VTK"""
    gmsh.initialize()
    gmsh.open(mesh_path)
    
    nodes = gmsh.model.mesh.getNodes()
    elements = gmsh.model.mesh.getElements()
    
    # Get triangular elements
    triangle_elements = None
    for i, type_id in enumerate(elements[0]):
        if type_id == 2:  # Triangle type
            triangle_elements = elements[1][i]
            triangle_nodes = elements[2][i]
            break
    
    if triangle_elements is None:
        raise ValueError("No triangular elements found in mesh")
    
    # Create VTK points
    points = vtk.vtkPoints()
    for i in range(0, len(nodes[0])):
        points.InsertNextPoint(
            float(nodes[1][i*3]),
            float(nodes[1][i*3 + 1]),
            float(nodes[1][i*3 + 2])
        )
    
    # Create triangular cells with explicit type conversion
    cells = vtk.vtkCellArray()
    for i in range(0, len(triangle_nodes), 3):
        triangle = vtk.vtkTriangle()
        # Convert to integer indices
        idx0 = int(triangle_nodes[i] - 1)
        idx1 = int(triangle_nodes[i + 1] - 1)
        idx2 = int(triangle_nodes[i + 2] - 1)
        triangle.GetPointIds().SetId(0, idx0)
        triangle.GetPointIds().SetId(1, idx1)
        triangle.GetPointIds().SetId(2, idx2)
        cells.InsertNextCell(triangle)
    
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.SetCells(vtk.VTK_TRIANGLE, cells)
    
    mesh = vedo.UnstructuredGrid(grid)
    gmsh.finalize()
    return mesh


def plot_mesh_with_error(mesh_path, points, errors):
    mesh = load_mesh_safe(mesh_path)
    mesh.cmap('viridis', errors)
    mesh.add_scalarbar(title='Error [m]')
    
    plotter = vedo.Plotter(axes=1, bg='white')
    plotter += mesh
    plotter += vedo.Points(points, r=3, c='black', alpha=0.5)
    
    return plotter

# Usage
mesh_path = "mesh/lego_brick_3867.msh"
plotter = plot_mesh_with_error(mesh_path, gt_grid, error_gt)
plotter.show()