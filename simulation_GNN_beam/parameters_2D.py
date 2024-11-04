"""
Parameters
Define the following set of parameters :
    * grid parameters
"""

from numpy import array
from collections import namedtuple

nx = 50
ny = 20
nz = 1 # 2D simulation


# Grid parameters
grid_min = array([0., -1., 0.])
grid_max = array([10., 1., 0.])
grid_resolution = array([nx, ny, nz])
grid_nb_nodes = grid_resolution[0] * grid_resolution[1] * grid_resolution[2]
grid_fixed_box = array([-0.1, -2.1, -0.1, 0.1, 2.1, 0.1])
grid = {'min': grid_min,
        'max': grid_max,
        'res': grid_resolution,
        'size': grid_min.tolist() + grid_max.tolist(),
        'nb_nodes': grid_nb_nodes,
        'fixed_box': grid_fixed_box}
p_grid = namedtuple('p_grid', grid)(**grid)

nx_LR = 15
# nx_LR = nx // 2
ny_LR = 5
# ny_LR = ny // 2
nz_LR = nz

# Low resolution grid definition
grid_min_LR = array([0., -1., 0.])
grid_max_LR = array([10., 1., 0.])
grid_resolution_LR = array([nx_LR, ny_LR, nz_LR])
grid_nb_nodes_LR = grid_resolution_LR[0] * grid_resolution_LR[1] * grid_resolution_LR[2]
grid_fixed_box_LR = array([-0.1, -2.1, -0.1, 0.1, 2.1, 0.1])
grid_LR = {'min': grid_min_LR,
           'max': grid_max_LR,
           'res': grid_resolution_LR,
           'size': grid_min_LR.tolist() + grid_max_LR.tolist(),
           'nb_nodes': grid_nb_nodes_LR,
           'fixed_box': grid_fixed_box_LR}
p_grid_LR = namedtuple('p_grid_LR', grid_LR)(**grid_LR)

#pretty print grid parameters
def print_grid_parameters():
    print("Grid parameters:")
    for key, value in grid.items():
        print(f"{key}: {value}")
    print("\nLow resolution grid parameters:")
    for key, value in grid_LR.items():
        print(f"{key}: {value}")
    print("\n")

if __name__ == '__main__':
    print_grid_parameters()
    
