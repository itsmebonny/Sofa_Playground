import Sofa
import SofaRuntime
import numpy as np 
import os
from Sofa import SofaDeformable
from time import process_time, time
import datetime
from sklearn.preprocessing import MinMaxScaler
from parameters_2D import p_grid, p_grid_LR
# add network path to the python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../network'))

import json

from parameters_2D import p_grid, p_grid_LR



class AnimationStepController(Sofa.Core.Controller):
    def __init__(self, node, filename_high, filename_low, directory, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.externalForce = [0, 5, 0]
        self.object_mass = 0.5
        self.createGraph(node, filename_high, filename_low)
        self.root = node
        self.save = False
        self.l2_error, self.MSE_error = [], []
        self.l2_deformation, self.MSE_deformation = [], []
        self.RMSE_error, self.RMSE_deformation = [], []
        self.directory = directory
        print(f"Using directory: {self.directory}")
        print(f"Using filename_high: {filename_high}")
        print(f"Using filename_low: {filename_low}")


    def createGraph(self, rootNode, filename_high, filename_low):
        self.filename_high = filename_high
        self.filename_low = filename_low

        rootNode.addObject('RequiredPlugin', name='MultiThreading')
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Projective') # Needed to use components [FixedProjectiveConstraint]  
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.Engine.Select') # Needed to use components [BoxROI]  
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Iterative') # Needed to use components [CGLinearSolver]  
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Direct')
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mass')
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mapping.Linear') # Needed to use components [BarycentricMapping]  
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.MechanicalLoad') # Needed to use components [ConstantForceField]  
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.ODESolver.Backward') # Needed to use components [StaticSolver]  
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.FEM.Elastic') # Needed to use components [TriangularFEMForceFieldOptim]  
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.StateContainer') # Needed to use components [MechanicalObject]  
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Dynamic') # Needed to use components [TriangleSetTopologyContainer]  
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Grid') # Needed to use components [RegularGridTopology]  
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.Visual') # Needed to use components [VisualStyle]  


        rootNode.addObject('DefaultAnimationLoop')
        rootNode.addObject('DefaultVisualManagerLoop') 
        rootNode.addObject('VisualStyle', name="visualStyle", displayFlags="showBehaviorModels showCollisionModels")



        
        sphereRadius=0.025


        # Define material properties
        young_modulus = 5000
        poisson_ratio = 0.25

        self.coarse = rootNode.addChild('SamplingNodes')
        self.coarse.addObject('RegularGridTopology', name='coarseGridHigh', min=p_grid.min, max=p_grid.max, nx=p_grid.res[0], ny=p_grid.res[1], nz=p_grid.res[2])
        self.coarse.addObject('TetrahedronSetTopologyContainer', name='triangleTopoHigh', src='@coarseGridHigh')
        self.MO_sampling = self.coarse.addObject('MechanicalObject', name='coarseDOFsHigh', template='Vec3d', src='@coarseGridHigh')
        self.coarse.addObject('SphereCollisionModel', radius=1e-8, group=1, color='1 0 0')
        #self.coarse.addObject('BarycentricMapping', name="mapping", input='@DOFs', input_topology='@triangleTopo', output='@coarseDOFsHigh', output_topology='@triangleTopoHigh')



        self.exactSolution = rootNode.addChild('HighResSolution2D', activated=True)
        self.exactSolution.addObject('MeshGmshLoader', name='grid', filename=filename_high)
        self.surface_topo = self.exactSolution.addObject('TetrahedronSetTopologyContainer', name='triangleTopo', src='@grid')
        self.MO1 = self.exactSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        # self.exactSolution.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@quadTopo")
        self.solver = self.exactSolution.addObject('StaticSolver', name='ODE', newton_iterations="30", printLog=True)
        self.exactSolution.addObject('ParallelCGLinearSolver', template="ParallelCompressedRowSparseMatrixMat3x3d", iterations=1000, tolerance=1e-10, threshold=1e-10, warmStart=True)
        self.exactSolution.addObject('ParallelTetrahedronFEMForceField', name="FEM", youngModulus=young_modulus, poissonRatio=poisson_ratio, method="large", updateStiffnessMatrix="false")
        self.exactSolution.addObject('BoxROI', name='ROI', box="-0.1 -0.1 -0.1 0.1 5.1 1.1", drawBoxes=True)
        self.exactSolution.addObject('FixedConstraint', indices="@ROI.indices")
        self.cff_box = self.exactSolution.addObject('BoxROI', name='ROI2', box="9.9 -0.1 -0.1 10.1 5.1 1.1", drawBoxes=True)
        self.cff = self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")

        self.mapping = self.exactSolution.addChild("SamplingMapping")
        self.MO_MapHR = self.mapping.addObject('MechanicalObject', name='DOFs_HR', template='Vec3d', src='@../../SamplingNodes/coarseGridHigh')
        self.mapping.addObject('BarycentricMapping', name="mapping", input='@../DOFs', input_topology='@../triangleTopo', output='@DOFs_HR')
        self.mapping.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 0')



        self.exactSolution.addChild("visual")
        self.exactSolution.visual.addObject('OglModel', src='@../DOFs', color='0 1 0 1')
        self.exactSolution.visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')

        # same object with different resolution

        self.LowResSolution = rootNode.addChild('LowResSolution2D', activated=True)
        self.LowResSolution.addObject('MeshGmshLoader', name='gridLow', filename=filename_low)
        self.surface_topo_LR = self.LowResSolution.addObject('TetrahedronSetTopologyContainer', name='quadTopo', src='@gridLow')
        self.MO2 = self.LowResSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@gridLow')
        # self.LowResSolution.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@quadTopo")
        self.solver_LR = self.LowResSolution.addObject('StaticSolver', name='ODE', newton_iterations="30", printLog=True)
        self.LowResSolution.addObject('ParallelCGLinearSolver', template="ParallelCompressedRowSparseMatrixMat3x3d", iterations=500, tolerance=1e-08, threshold=1e-08, warmStart=False) 
        self.LowResSolution.addObject('ParallelTetrahedronFEMForceField', name="FEM", youngModulus=5000, poissonRatio=0.4, method="large", updateStiffnessMatrix="false")
        self.LowResSolution.addObject('BoxROI', name='ROI', box="-0.1 -0.1 -0.1 0.1 5.1 1.1", drawBoxes=True)
        self.LowResSolution.addObject('FixedConstraint', indices="@ROI.indices")
        self.cff_box_LR = self.LowResSolution.addObject('BoxROI', name='ROI2', box="9.9 -0.1 -0.1 10.1 2.1 1.1", drawBoxes=True)
        self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")

        self.mapping = self.LowResSolution.addChild("SamplingMapping")
        self.MO_MapLR = self.mapping.addObject('MechanicalObject', name='DOFs_HR', template='Vec3d', src='@../../SamplingNodes/coarseGridHigh')
        self.mapping.addObject('BarycentricMapping', name="mapping", input='@../DOFs', input_topology='@../triangleTopo', output='@DOFs_HR')
        self.mapping.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 1')



        self.LowResSolution.addChild("visual")
        self.visual_model = self.LowResSolution.visual.addObject('OglModel', src='@../DOFs', color='0 1 1 1')
        self.LowResSolution.visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')

        self.nb_nodes = len(self.MO1.position.value)
        self.nb_nodes_LR = len(self.MO2.position.value)
        


    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        """
        self.inputs = []
        self.outputs = []
        self.save = True
        self.efficient_sampling = False
        self.iteration = 0
        if self.efficient_sampling:
            self.count_v = 0
            self.num_versors = 5
            self.versors = self.generate_versors(self.num_versors)
            self.magnitudes = np.linspace(0, 80, 30)
            self.count_m = 0
            self.angles = np.linspace(0, 2*np.pi, self.num_versors, endpoint=False)
            self.starting_points = np.linspace(self.angles[0], self.angles[1], len(self.magnitudes), endpoint=False)
        if self.save:
            if not os.path.exists('npy_GNN_lego'):
                os.mkdir('npy_GNN_lego')
            # get current time from computer format yyyy-mm-dd-hh-mm-ss and create a folder with that name
            if not os.path.exists(f'npy_GNN_lego/{self.directory}'):
                os.makedirs(f'npy_GNN_lego/{self.directory}')
            print(f"Saving data to npy_GNN_lego/{self.directory}")
        self.sampled = False

        surface = self.surface_topo
        surface_LR = self.surface_topo_LR

        self.idx_surface = surface.triangles.value.reshape(-1)
        self.idx_surface_LR = surface_LR.triangles.value.reshape(-1)

        print(f"Number of nodes in the high resolution solution: {self.nb_nodes}")
        print(f"Number of nodes in the low resolution solution: {self.nb_nodes_LR}")
        print(f"Number of nodes in the sampling grid: {len(self.MO_sampling.position.value)}")
        print(f"Number of nodes in the high resolution mapping: {len(self.MO_MapHR.position.value)}")
        print(f"Number of nodes in the low resolution mapping: {len(self.MO_MapLR.position.value)}")


    def onAnimateBeginEvent(self, event):
    
        self.bad_sample = False
        # reset positions
        self.MO1.position.value = self.MO1.rest_position.value
        self.MO2.position.value = self.MO2.rest_position.value
   
        if not self.efficient_sampling:
            self.z = np.random.uniform(-1, 1)
            self.phi = np.random.uniform(0, 2*np.pi)
            self.versor = np.array([np.sqrt(1 - self.z**2) * np.cos(self.phi), np.sqrt(1 - self.z**2) * np.sin(self.phi), self.z])
            self.magnitude = np.random.uniform(0, 25)
            self.externalForce = self.magnitude * self.versor
        else:
            self.sample = self.count_m *self.num_versors + self.count_v
            self.externalForce = np.append(self.magnitudes[self.count_m] * self.versors[self.count_v], 0)
            
            self.count_v += 1
            if self.count_v == len(self.versors):
                self.count_v = 0
                self.count_m += 1
                self.versors = self.generate_versors(self.num_versors, starting_point=self.starting_points[self.count_m])
            if self.count_m == len(self.magnitudes):
                self.count_m = 0
                self.count_v = 0
                self.sampled = True




       # Define random box
        side = np.random.randint(1, 6)
        print(f"Side: {side}")
        if side == 1:
            x_min = np.random.uniform(2, 8.0)
            x_max = x_min + 2
            y_min = np.random.uniform(0, 3.0)
            y_max = y_min + 2
            z_min = -0.1
            z_max = 0.1
        elif side == 2:
            x_min = np.random.uniform(2, 8.0)
            x_max = x_min + 2
            y_min = np.random.uniform(-1, 3.0)
            y_max = y_min + 2
            z_min = 0.99
            z_max = 1.01
        elif side == 3:
            x_min = np.random.uniform(2, 8.0)
            x_max = x_min + 2
            y_min = -0.1
            y_max = 0.1
            z_min = np.random.uniform(0, 0.5)
            z_max = z_min + 0.5
        elif side == 4:
            x_min = np.random.uniform(2, 8.0)
            x_max = x_min + 2
            y_min = 4.9
            y_max = 5.1
            z_min = np.random.uniform(0, 0.5)
            z_max = z_min + 0.5
        elif side == 5:
            x_min = 9.99
            x_max = 10.01
            y_min = np.random.uniform(0, 3.0)
            y_max = y_min + 2
            z_min = np.random.uniform(0, 0.5)
            z_max = z_min + 0.5

        


        
        #print(f"==================== Intersected squares: {intersect_count}  with magnitude {self.magnitude}====================")

        bbox = [x_min, y_min, z_min, x_max, y_max, z_max]

        self.exactSolution.removeObject(self.cff_box)
        self.cff_box = self.exactSolution.addObject('BoxROI', name='ROI2', box=bbox, drawBoxes=True)
        self.cff_box.init()

        self.LowResSolution.removeObject(self.cff_box_LR)
        self.cff_box_LR = self.LowResSolution.addObject('BoxROI', name='ROI2', box=bbox, drawBoxes=True)
        self.cff_box_LR.init()

        # Get the intersection with the surface
        indices = list(self.cff_box.indices.value)
        indices = list(set(indices).intersection(set(self.idx_surface)))
        print(f"Number of nodes in the high resolution solution: {len(indices)}")
        indices_LR = list(self.cff_box_LR.indices.value)
        indices_LR = list(set(indices_LR).intersection(set(self.idx_surface_LR)))
        print(f"Number of nodes in the low resolution solution: {len(indices_LR)}")
        self.exactSolution.removeObject(self.cff)
        self.cff = self.exactSolution.addObject('ConstantForceField', indices=indices, totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff.init()


        self.LowResSolution.removeObject(self.cffLR)
        self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices=indices_LR, totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cffLR.init()


        indices_training = np.where((self.MO_MapLR.rest_position.value[:, 0] >= x_min) & (self.MO_MapLR.rest_position.value[:, 0] <= x_max) & (self.MO_MapLR.rest_position.value[:, 1] >= y_min) & (self.MO_MapLR.rest_position.value[:, 1] <= y_max) & (self.MO_MapLR.rest_position.value[:, 2] >= z_min) & (self.MO_MapLR.rest_position.value[:, 2] <= z_max))[0]

        #print(f"Number of nodes in the training set: {len(indices_training)}")

        self.bounding_box = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "z_min": z_min, "z_max": z_max}
        self.versor_rounded = [round(i, 4) for i in self.versor]
        self.versor_rounded = list(self.versor_rounded)
        self.force_info = {"magnitude": round(self.magnitude, 4), "versor": self.versor_rounded}
        self.indices_BC = list(indices_training)
        for i in range(len(indices_training)):
            self.indices_BC[i] = int(indices_training[i])

        #print(f"Bounding box: [{x_min}, {y_min}, {z_min}, {x_max}, {y_max}, {z_max}]")
        #print(f"Side: {side}")
        if indices_training.size == 0:
            #print("Empty intersection")
            self.bad_sample = True
        self.start_time = process_time()
       



    def onAnimateEndEvent(self, event):
        self.end_time = process_time()
        U_high = self.compute_displacement(self.MO_MapHR)
        U_low = self.compute_displacement(self.MO_MapLR)
        U = self.compute_displacement(self.MO1)
        edges_low = self.compute_edges(self.surface_topo_LR)
       
        if self.save and self.bad_sample == False:
            self.tmp_dir = f'npy_GNN_lego/{self.directory}/sample_{self.iteration}'
            if not os.path.exists(self.tmp_dir):
                os.makedirs(self.tmp_dir)
            else:
                print("Directory already exists, something went wrong")

            np.save(f'{self.tmp_dir}/high_res_displacement.npy', U_high)
            np.save(f'{self.tmp_dir}/low_res_displacement.npy', U_low)
            np.save(f'{self.tmp_dir}/edges_low.npy', edges_low)
            np.save(f'{self.tmp_dir}/exact_displacement.npy', U)
            #save in a JSON file the bounding box and the force info with the structure Iteration -> Bounding box -> Force info and close the file
            with open(f'{self.tmp_dir}/info.json', 'w') as f:
                json.dump({'iteration': self.iteration, 'bounding_box': self.bounding_box, 'force_info': self.force_info, 'indices_BC': self.indices_BC}, f)

            
            #print(f"Saved data to {self.tmp_dir}")
            self.iteration += 1


    def compute_displacement(self, mechanical_object):
        # Compute the displacement between the high and low resolution solutions
        U = mechanical_object.position.value.copy() - mechanical_object.rest_position.value.copy()
        return U
    
    def compute_velocity(self, mechanical_object):
        # Compute the velocity of the high resolution solution
        return mechanical_object.velocity.value.copy()
    

    def compute_rest_position(self, mechanical_object):
        # Compute the position of the high resolution solution
        return mechanical_object.rest_position.value.copy()
    
    def compute_edges(self, grid):
        # create a matrix with the edges of the grid and their length
        edges = grid.edges.value.copy()
        positions = grid.position.value.copy()
        matrix = np.zeros((len(edges)*2, 3))
        for i, edge in enumerate(edges):
            # account for the fact the edges must be bidirectional
            matrix[len(edges) + i, 0] = edge[1]
            matrix[len(edges) + i, 1] = edge[0]
            matrix[len(edges) + i, 2] = np.linalg.norm(positions[edge[0]] - positions[edge[1]])
            matrix[i, 0] = edge[0]
            matrix[i, 1] = edge[1]
            matrix[i, 2] = np.linalg.norm(positions[edge[0]] - positions[edge[1]])
        return matrix
    
    def generate_versors(self, n_versors):
            # Initialize an empty list to store the versors
            a = 4*np.pi/n_versors
            d = np.sqrt(a)
            M_theta = int(np.round(np.pi/d))
            d_theta = np.pi/M_theta
            d_phi = a/d_theta
            versors = []
            for m in range(M_theta):
                theta = np.pi*(m + 0.5)/M_theta
                M_phi = int(np.round(2*np.pi*np.sin(theta)/d_phi))
                for n in range(M_phi):
                    phi = 2*np.pi*n/M_phi
                    versors.append([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
            return np.array(versors)
    
    def close(self):
        print("Closing simulation")


def createScene(rootNode, filename_high, filename_low, directory, *args, **kwargs):
    rootNode.dt = 0.01
    rootNode.gravity = [0, 0, 0]
    rootNode.name = 'root'
    asc = AnimationStepController(rootNode, filename_high, filename_low, directory, *args, **kwargs)
    rootNode.addObject(asc)
    return rootNode, asc


def main():
    import Sofa.Gui
    from tqdm import tqdm
    import re
    SofaRuntime.importPlugin("Sofa.GL.Component.Rendering3D")
    SofaRuntime.importPlugin("Sofa.GL.Component.Shader")
    SofaRuntime.importPlugin("Sofa.Component.StateContainer")
    SofaRuntime.importPlugin("Sofa.Component.ODESolver.Backward")
    SofaRuntime.importPlugin("Sofa.Component.LinearSolver.Direct")
    SofaRuntime.importPlugin("Sofa.Component.IO.Mesh")
    SofaRuntime.importPlugin("Sofa.Component.MechanicalLoad")
    SofaRuntime.importPlugin("Sofa.Component.Engine.Select")
    SofaRuntime.importPlugin("Sofa.Component.SolidMechanics.FEM.Elastic")

    

    def process_mesh_files(mesh_dir):
        # Get all .msh files
        mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.msh')]
        mesh_dict = {}
        
        # Extract parameters from filenames and group files
        for file in mesh_files:
            # Extract x, y, r, n values using regex
            match = re.search(r'plate_x(\d+\.?\d*)_y(\d+\.?\d*)_r(\d+\.?\d*)_n(\d+)\.msh', file)
            if match:
                x, y, r, n = map(float, match.groups())
                key = (x, y, r)
                
                if key not in mesh_dict:
                    mesh_dict[key] = []
                mesh_dict[key].append((int(n), os.path.join('mesh', file)))
        # Sort by node count and keep only pairs
        sorted_dict = {}
        for key, files in mesh_dict.items():
            if len(files) == 2:
                # Sort by number of nodes
                sorted_files = sorted(files, key=lambda x: x[0])
                # Store only file paths in sorted order
                sorted_dict[key] = [f[1] for f in sorted_files]
        
        return sorted_dict
    USE_GUI = True
    # Update main code
    files = process_mesh_files('mesh')
    roots_train = []
    roots_val = []
    ascs_train = []
    ascs_val = []
    mode = "train"
    directory = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    training_dir = directory + "_training"
    validation_dir = directory + "_validation"
    mesh_training = 10
    training_samples = 100
    validation_samples = 20
    #shuffle the files
    keys = list(files.keys())
    np.random.shuffle(keys)
    files = {k: files[k] for k in keys}
    for i, (key, values) in enumerate(files.items()):
        
        
        if i < mesh_training:
            root = Sofa.Core.Node("root")
            rootNode, asc = createScene(root, values[1], values[0], training_dir)
            Sofa.Simulation.init(root)
            if not USE_GUI:
                for i in tqdm(range(training_samples)):
                        Sofa.Simulation.animate(root, root.dt.value)
            else:
                Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
                Sofa.Gui.GUIManager.createGUI(root, __file__)
                Sofa.Gui.GUIManager.SetDimension(800, 600)
                Sofa.Gui.GUIManager.MainLoop(root)
                Sofa.Gui.GUIManager.closeGUI()

        else:
            root = Sofa.Core.Node("root")
            root, asc = createScene(root, values[1], values[0], validation_dir)
            Sofa.Simulation.init(root)
            for i in tqdm(range(validation_samples)):
                    Sofa.Simulation.animate(root, root.dt.value)
                    Sofa.Simulation.reset(root)





    
    if mode == "train":
        directory = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        directory = directory + "_training"
        
        for root, asc in zip(roots_train, ascs_train):
            Sofa.Simulation.init(root)
            

           
    else:
        
        for root, asc in zip(roots_val, ascs_val):
            Sofa.Simulation.init(root)
            if not USE_GUI:
                for i in tqdm(range(validation_samples)):
                    Sofa.Simulation.animate(root, root.dt.value)
                    Sofa.Simulation.reset(root)
            else:
                Sofa.Simulation.init(root)
                Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
                Sofa.Gui.GUIManager.createGUI(root, __file__)
                Sofa.Gui.GUIManager.SetDimension(800, 600)
                Sofa.Gui.GUIManager.MainLoop(root)
                Sofa.Gui.GUIManager.closeGUI()
                Sofa.Simulation.reset(root)


if __name__ == '__main__':
    main()
