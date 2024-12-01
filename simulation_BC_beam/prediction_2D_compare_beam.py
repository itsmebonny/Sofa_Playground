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
import torch as th

import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../network_BC'))

print(sys.path)

from network_BC.GNN_Error_estimation import Trainer as Trainer
from network_BC.FC_Error_estimation_integrated import Trainer as TrainerFC
from parameters_2D import p_grid, p_grid_LR
from torch_geometric.data import Data

from scipy.interpolate import RBFInterpolator, griddata, RegularGridInterpolator, interpn

#########################################
# CORREGGI LA FORZA APPLICATA PERCHE' NON E' CORRETTA
#########################################


class AnimationStepController(Sofa.Core.Controller):
    def __init__(self, node, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.externalForce = [0, 30, 0]
        self.object_mass = 0.5
        self.createGraph(node)
        self.root = node
        self.save = False
        self.l2_error, self.MSE_error = [], []
        self.l2_deformation, self.MSE_deformation = [], []
        self.RMSE_error, self.RMSE_deformation = [], []
        self.l2_error_FC, self.MSE_error_FC = [], []
        self.l2_deformation_FC, self.MSE_deformation_FC = [], []
        self.RMSE_error_FC, self.RMSE_deformation_FC = [], []
        self.save_for_images = False

        self.network = Trainer('npy_BC_beam/2024-11-23_16:54:46_fast_loading', 32, 0.001, 500)
        self.network.load_model('models_BC/model_2024-11-21_10:53:27_GNN_beam.pth')

        self.networkFC = TrainerFC('npy_BC_beam/2024-11-23_16:54:46_fast_loading', 32, 0.001,  500)
        self.networkFC.load_model('models_BC/model_2024-11-23_13:28:54_FC_beam.pth')
        
    def createGraph(self, rootNode):

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


        filename_high = 'mesh/beam_5080.msh'
        filename_low = 'mesh/beam_410.msh'
        stl_filename = 'mesh/beam.stl'

        self.loader = rootNode.addObject('MeshSTLLoader', name='loader', filename=stl_filename)

        self.coarse = rootNode.addChild('SamplingNodes')
        self.coarse.addObject('RegularGridTopology', name='coarseGridHigh', min=p_grid.min, max=p_grid.max, nx=p_grid.res[0], ny=p_grid.res[1], nz=p_grid.res[2])
        self.coarse.addObject('TetrahedronSetTopologyContainer', name='triangleTopoHigh', src='@coarseGridHigh')
        self.MO_sampling = self.coarse.addObject('MechanicalObject', name='coarseDOFsHigh', template='Vec3d', src='@coarseGridHigh')
        self.coarse.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='1 0 0')
        #self.coarse.addObject('BarycentricMapping', name="mapping", input='@DOFs', input_topology='@triangleTopo', output='@coarseDOFsHigh', output_topology='@triangleTopoHigh')


        self.exactSolution = rootNode.addChild('HighResSolution2D', activated=True)
        self.exactSolution.addObject('MeshGmshLoader', name='grid', filename=filename_high)
        self.surface_topo = self.exactSolution.addObject('TetrahedronSetTopologyContainer', name='triangleTopo', src='@grid')
        self.MO1 = self.exactSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        # self.exactSolution.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@quadTopo")
        self.solver = self.exactSolution.addObject('StaticSolver', name='ODE', newton_iterations="20", printLog=True)
        self.exactSolution.addObject('ParallelCGLinearSolver', template="ParallelCompressedRowSparseMatrixMat3x3d", iterations=2500, tolerance=1e-08, threshold=1e-08, warmStart=True)
        self.exactSolution.addObject('ParallelTetrahedronFEMForceField', name="FEM", youngModulus=5000, poissonRatio=0.4, method="large", updateStiffnessMatrix="false")
        self.exactSolution.addObject('BoxROI', name='ROI', box="-0.1 -1.1 -1.1 0.1 1.1 1.1", drawBoxes=True)
        self.exactSolution.addObject('FixedConstraint', indices="@ROI.indices")
        self.cff_box = self.exactSolution.addObject('BoxROI', name='ROI2', box="9.9 -1.1 -1.1 10.1 1.1 1.1", drawBoxes=True)
        self.cff = self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")

        self.mapping = self.exactSolution.addChild("SamplingMapping")
        self.MO_MapHR = self.mapping.addObject('MechanicalObject', name='DOFs_HR', template='Vec3d', src='@../../SamplingNodes/coarseGridHigh')
        #self.MO1_HR = self.mapping.addObject('MechanicalObject', name='DOFs_HR', template='Vec3d', position='1 3 0')
        self.mapping.addObject('BarycentricMapping', name="mapping", input='@../DOFs', input_topology='@../triangleTopo', output='@DOFs_HR')
        self.mapping.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 0')

        self.exactSolution.addChild("visual")
        self.exactSolution.visual.addObject('OglModel', src='@../grid', color='0 1 1 0.5')
        self.exactSolution.visual.addObject('IdentityMapping', input='@../DOFs', output='@./')

        # same object with different resolution

        self.LowResSolution = rootNode.addChild('LowResSolution2D', activated=True)
        self.LowResSolution.addObject('MeshGmshLoader', name='gridLow', filename=filename_low)
        self.surface_topo_LR = self.LowResSolution.addObject('TetrahedronSetTopologyContainer', name='quadTopo', src='@gridLow')
        self.MO2 = self.LowResSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@gridLow')
        # self.LowResSolution.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@quadTopo")
        self.solver_LR = self.LowResSolution.addObject('StaticSolver', name='ODE', newton_iterations="10", printLog=True)
        self.LowResSolution.addObject('ParallelCGLinearSolver', template="ParallelCompressedRowSparseMatrixMat3x3d", iterations=2000, tolerance=1e-08, threshold=1e-08, warmStart=True) 
        self.LowResSolution.addObject('ParallelTetrahedronFEMForceField', name="FEM", youngModulus=5000, poissonRatio=0.4, method="large", updateStiffnessMatrix="false")
        self.LowResSolution.addObject('BoxROI', name='ROI', box="-0.1 -1.1 -1.1 0.1 1.1 1.1", drawBoxes=True)
        self.LowResSolution.addObject('FixedConstraint', indices="@ROI.indices")
        self.cff_box_LR = self.LowResSolution.addObject('BoxROI', name='ROI2', box="9.9 -1.1 -1.1 10.1 1.1 1.1", drawBoxes=True)
        self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")

        self.mapping = self.LowResSolution.addChild("SamplingMapping")
        self.MO_MapLR = self.mapping.addObject('MechanicalObject', name='DOFs_HR', template='Vec3d', src='@../../SamplingNodes/coarseGridHigh')
        #self.MO1_LR = self.mapping.addObject('MechanicalObject', name='DOFs_HR', template='Vec3d', position='1 3 0')
        self.mapping.addObject('BarycentricMapping', name="mapping", input='@../DOFs', input_topology='@../triangleTopo', output='@DOFs_HR')
        self.mapping.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 1')

        self.LowResSolution.addChild("visual")
        self.visual_model = self.LowResSolution.visual.addObject('OglModel', src='@../gridLow', color='1 0 0 0.5')
        self.LowResSolution.visual.addObject('IdentityMapping', input='@../DOFs', output='@./')


        self.LowResSolution_FC = rootNode.addChild('LowResSolution2D_FC', activated=True)


        self.LowResSolution_FC.addObject('MeshGmshLoader', name='gridLow', filename=filename_low)
        self.surface_topo_LR_FC = self.LowResSolution_FC.addObject('TetrahedronSetTopologyContainer', name='quadTopo', src='@gridLow')
        self.MO2_FC = self.LowResSolution_FC.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@gridLow')
        self.solver_LR_FC = self.LowResSolution_FC.addObject('StaticSolver', name='ODE', newton_iterations="10", printLog=True)
        self.LowResSolution_FC.addObject('ParallelCGLinearSolver', template="ParallelCompressedRowSparseMatrixMat3x3d", iterations=2000, tolerance=1e-08, threshold=1e-08, warmStart=True) 
        self.LowResSolution_FC.addObject('ParallelTetrahedronFEMForceField', name="FEM", youngModulus=5000, poissonRatio=0.4, method="large", updateStiffnessMatrix="false")
        self.LowResSolution_FC.addObject('BoxROI', name='ROI', box="-0.1 -1.1 -1.1 0.1 1.1 1.1", drawBoxes=True)
        self.LowResSolution_FC.addObject('FixedConstraint', indices="@ROI.indices")
        self.cff_box_LR_FC = self.LowResSolution_FC.addObject('BoxROI', name='ROI2', box="9.9 -1.1 -1.1 10.1 1.1 1.1", drawBoxes=True)
        self.cffLR_FC = self.LowResSolution_FC.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")

        self.mapping_FC = self.LowResSolution_FC.addChild("SamplingMapping")
        self.MO_MapLR_FC = self.mapping_FC.addObject('MechanicalObject', name='DOFs_HR', template='Vec3d', src='@../../SamplingNodes/coarseGridHigh')
        self.mapping_FC.addObject('BarycentricMapping', name="mapping", input='@../DOFs', input_topology='@../triangleTopo', output='@DOFs_HR')
        self.mapping_FC.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 1')

        self.LowResSolution_FC.addChild("visual")
        self.visual_model_FC = self.LowResSolution_FC.visual.addObject('OglModel', src='@../gridLow', color='1 1 0 0.5')
        self.LowResSolution_FC.visual.addObject('IdentityMapping', input='@../DOFs', output='@./')


        self.nb_nodes = len(self.MO1.position.value)
        self.nb_nodes_LR = len(self.MO_MapLR.position.value)

        


    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        """
        self.inputs = []
        self.outputs = []
        self.efficient_sampling = False
        if self.efficient_sampling:
            self.count_v = 0
            self.num_versors = 5
            self.versors = self.generate_versors(self.num_versors)
            self.magnitudes = np.linspace(0, 80, 30)
            self.count_m = 0
            self.angles = np.linspace(0, 2*np.pi, self.num_versors, endpoint=False)
            self.starting_points = np.linspace(self.angles[0], self.angles[1], len(self.magnitudes), endpoint=False)
        if self.save:
            if not os.path.exists('npy_GNN_beam'):
                os.mkdir('npy_GNN_beam')
            # get current time from computer format yyyy-mm-dd-hh-mm-ss and create a folder with that name
            self.directory = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            self.directory = self.directory + "_estimation"
            if self.efficient_sampling:
                self.directory = self.directory + "_efficient"
            os.makedirs(f'npy_GNN_beam/{self.directory}')
            print(f"Saving data to npy_GNN/{self.directory}")
        self.sampled = False

        surface = self.surface_topo
        surface_LR = self.surface_topo_LR
        surface_FC = self.surface_topo_LR_FC

        self.idx_surface = surface.triangles.value.reshape(-1)
        self.idx_surface_LR = surface_LR.triangles.value.reshape(-1)
        self.idx_surface_FC = surface_FC.triangles.value.reshape(-1)

        self.low_res_shape = self.MO_MapLR.position.value.shape
        self.high_res_shape = self.MO1.position.value.shape

        print(f"Number of nodes in the high resolution solution: {self.nb_nodes}")
        print(f"Number of nodes in the low resolution solution: {self.nb_nodes_LR}")
        print(f"Low resolution shape: {self.low_res_shape}")
        print(f"High resolution shape: {self.high_res_shape}")

    def onAnimateBeginEvent(self, event):
    
        self.bad_sample = False
        # reset positions
        self.MO1.position.value = self.MO1.rest_position.value
        self.MO2.position.value = self.MO2.rest_position.value
        self.MO2_FC.position.value = self.MO2_FC.rest_position.value

        if self.sampled:
            print("================== Sampled all magnitudes and versors ==================\n")
            print ("================== The simulation is over ==================\n")
        
        if not self.efficient_sampling:
            # self.z = np.random.uniform(-1, 1)
            # self.phi = np.random.uniform(0, 2*np.pi)
            # self.versor = np.array([np.sqrt(1 - self.z**2) * np.cos(self.phi), np.sqrt(1 - self.z**2) * np.sin(self.phi), self.z])
            # self.magnitude = np.random.uniform(0, 100)
            # self.externalForce = self.magnitude * self.versor
            self.theta = np.random.uniform(0, 2*np.pi)
            self.versor = np.array([np.cos(self.theta), np.sin(self.theta), 0])
            self.magnitude = np.random.uniform(0, 70)
            self.externalForce = np.append(self.magnitude * self.versor, 0)
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
        if side == 1:
            x_min = np.random.uniform(2, 9.0)
            x_max = x_min + 1
            y_min = np.random.uniform(-1, 0.0)
            y_max = y_min + 1
            z_min = -1.01
            z_max = -0.99
        elif side == 2:
            x_min = np.random.uniform(2, 9.0)
            x_max = x_min + 1
            y_min = np.random.uniform(-1, 0.0)
            y_max = y_min + 1
            z_min = 0.99
            z_max = 1.01
        elif side == 3:
            x_min = np.random.uniform(2, 9.0)
            x_max = x_min + 1
            y_min = -1.01
            y_max = -0.99
            z_min = np.random.uniform(-1, 0.0)
            z_max = z_min + 1
        elif side == 4:
            x_min = np.random.uniform(2, 9.0)
            x_max = x_min + 1
            y_min = 0.99
            y_max = 1.01
            z_min = np.random.uniform(-1, 0.0)
            z_max = z_min + 1
        elif side == 5:
            x_min = 9.99
            x_max = 10.01
            y_min = np.random.uniform(-1, 0.0)
            y_max = y_min + 1
            z_min = np.random.uniform(-1, 0.0)
            z_max = z_min + 1
        

        bbox = [x_min, y_min, z_min, x_max, y_max, z_max]

        self.exactSolution.removeObject(self.cff_box)
        self.cff_box = self.exactSolution.addObject('BoxROI', name='ROI2', box=bbox, drawBoxes=True)
        self.cff_box.init()

        self.LowResSolution.removeObject(self.cff_box_LR)
        self.cff_box_LR = self.LowResSolution.addObject('BoxROI', name='ROI2', box=bbox, drawBoxes=True)
        self.cff_box_LR.init()

        self.LowResSolution_FC.removeObject(self.cff_box_LR_FC)
        self.cff_box_LR_FC = self.LowResSolution_FC.addObject('BoxROI', name='ROI2', box=bbox, drawBoxes=True)
        self.cff_box_LR_FC.init()


        # Get the intersection with the surface
        indices = list(self.cff_box.indices.value)
        indices = list(set(indices).intersection(set(self.idx_surface)))
        print(f"Number of nodes in the high resolution solution: {len(indices)}")
        indices_LR = list(self.cff_box_LR.indices.value)
        indices_LR = list(set(indices_LR).intersection(set(self.idx_surface_LR)))
        print(f"Number of nodes in the low resolution solution: {len(indices_LR)}")

        indices_FC = list(self.cff_box_LR_FC.indices.value)
        indices_FC = list(set(indices_FC).intersection(set(self.idx_surface_FC)))
        print(f"Number of nodes in the low resolution solution: {len(indices_FC)}")

        self.exactSolution.removeObject(self.cff)
        self.cff = self.exactSolution.addObject('ConstantForceField', indices=indices, totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff.init()

        indices_training = np.where((self.MO_MapLR.rest_position.value[:, 0] >= x_min) & (self.MO_MapLR.rest_position.value[:, 0] <= x_max) & (self.MO_MapLR.rest_position.value[:, 1] >= y_min) & (self.MO_MapLR.rest_position.value[:, 1] <= y_max) & (self.MO_MapLR.rest_position.value[:, 2] >= z_min) & (self.MO_MapLR.rest_position.value[:, 2] <= z_max))[0]


        self.LowResSolution.removeObject(self.cffLR)
        self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices=indices_LR, totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cffLR.init()

        self.LowResSolution_FC.removeObject(self.cffLR_FC)
        self.cffLR_FC = self.LowResSolution_FC.addObject('ConstantForceField', indices=indices_FC, totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cffLR_FC.init()


        self.bounding_box = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "z_min": z_min, "z_max": z_max}
        self.versor_rounded = [round(i, 4) for i in self.versor]
        print(f"External force: {self.versor_rounded}")
        self.versor_rounded = list(self.versor_rounded)
        self.force_info = {"magnitude": round(self.magnitude, 4), "versor": self.versor_rounded}
        self.indices_BC = list(indices_training)
        for i in range(len(indices_training)):
            self.indices_BC[i] = int(indices_training[i])

        print(f"Bounding box: [{x_min}, {y_min}, {z_min}, {x_max}, {y_max}, {z_max}]")

        print(f"Bounding box: [{x_min}, {y_min}, {z_min}, {x_max}, {y_max}, {z_max}]")
        print(f"Side: {side}")
        if indices_LR == [] or indices == []:
            print("Empty intersection")
            self.bad_sample = True
        self.start_time = process_time()



    def onAnimateEndEvent(self, event):
        

        self.end_time = process_time()
        print("Computation time for 1 time step: ", self.end_time - self.start_time)
        print("External force: ", np.linalg.norm(self.externalForce))
        U_low = self.compute_displacement(self.MO_MapLR)
        U_low_FC = self.compute_displacement(self.MO_MapLR_FC)
        edges_low = self.compute_edges(self.surface_topo_LR)
        print("U_low_FC: ", U_low_FC.shape)


        boundary_nodes = np.zeros((U_low.shape[0], 4))
        boundary_nodes[self.indices_BC, :3] = self.versor_rounded
        boundary_nodes[self.indices_BC, 3] = round(self.magnitude, 4)
        node_features = np.hstack ((U_low, boundary_nodes))
        edge_index = edges_low[:, :2].T
        edge_attr = edges_low[:, 2]
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

        U_low_FC = U_low_FC.flatten()
        x_min = self.bounding_box["x_min"]
        x_max = self.bounding_box["x_max"]
        y_min = self.bounding_box["y_min"]
        y_max = self.bounding_box["y_max"]
        z_min = self.bounding_box["z_min"]
        z_max = self.bounding_box["z_max"]
        boundary_conditions = np.zeros((10))
        boundary_conditions[:-4] = [x_min, x_max, y_min, y_max, z_min, z_max]
        boundary_conditions[-4:-1] = self.versor_rounded
        boundary_conditions[-1] = self.magnitude
        data_FC = np.append(U_low_FC, boundary_conditions)
        data_FC = th.tensor(data_FC, dtype=th.float32).T
        U = self.network.predict(data).cpu().numpy()
        U_FC = self.networkFC.predict(data_FC).cpu().numpy()
  

        

        U = np.reshape(U, (self.low_res_shape[0], self.low_res_shape[1]))
        U_FC = np.reshape(U_FC, (self.low_res_shape[0], self.low_res_shape[1]))


        self.MO_MapLR.position.value = self.MO_MapLR.position.value + U

        self.MO_MapLR_FC.position.value = self.MO_MapLR_FC.position.value + U_FC

                
        positions = self.MO_MapLR.rest_position.value.copy()
        
        displacement = self.MO_MapLR.position.value.copy() - self.MO_MapLR.rest_position.value.copy()

        interpolator = RBFInterpolator(positions, displacement, kernel='cubic')

        interpolate_positions = self.MO2.rest_position.value.copy()


        corrected_displacement = interpolator(interpolate_positions)
        
        self.MO2.position.value = interpolate_positions + corrected_displacement

        self.visual_model.position.value = interpolate_positions + corrected_displacement

        # FC MODEL INTERPOLATION

        positions_FC = self.MO_MapLR_FC.rest_position.value.copy()

        displacement_FC = self.MO_MapLR_FC.position.value.copy() - self.MO_MapLR_FC.rest_position.value.copy()

        interpolate_positions_FC = self.MO2_FC.rest_position.value.copy()
    

        corrected_displacement_FC = self.interpolate_vector_field(positions_FC, displacement_FC, interpolate_positions_FC)


        self.MO2_FC.position.value = interpolate_positions_FC + corrected_displacement_FC

        self.visual_model_FC.position.value = interpolate_positions_FC + corrected_displacement_FC


        if self.save_for_images:
            if not os.path.exists('images_data'):
                os.mkdir('images_data')

            if not os.path.exists(f'images_data/{self.directory}'):
                os.mkdir(f'images_data/{self.directory}')

            ground_truth_displacement = self.MO1.position.value - self.MO1.rest_position.value
            prediction_displacement = self.MO2.position.value - self.MO2.rest_position.value

            ground_truth_grid = self.MO1_LR.position.value - self.MO1_LR.rest_position.value
            prediction_grid = self.MO_training.position.value - self.MO_training.rest_position.value

            ground_truth_rest = self.MO1.rest_position.value
            prediction_rest = self.MO2.rest_position.value

            ground_truth_grid_rest = self.MO1_LR.rest_position.value
            prediction_grid_rest = self.MO_training.rest_position.value

            np.save(f'images_data/{self.directory}/ground_truth_displacement.npy', ground_truth_displacement)
            np.save(f'images_data/{self.directory}/prediction_displacement.npy', prediction_displacement)
            np.save(f'images_data/{self.directory}/ground_truth_grid.npy', ground_truth_grid)
            np.save(f'images_data/{self.directory}/prediction_grid.npy', prediction_grid)
            np.save(f'images_data/{self.directory}/ground_truth_rest.npy', ground_truth_rest)
            np.save(f'images_data/{self.directory}/prediction_rest.npy', prediction_rest)
            np.save(f'images_data/{self.directory}/ground_truth_grid_rest.npy', ground_truth_grid_rest)
            np.save(f'images_data/{self.directory}/prediction_grid_rest.npy', prediction_grid_rest)
            

        



        self.end_time = process_time()
      
        if not self.bad_sample:
            self.compute_metrics()
            self.compute_metrics_FC()
        print("Computation time for 1 time step: ", self.end_time - self.start_time)
        print("External force: ", np.linalg.norm(self.externalForce))

    def interpolate_vector_field(self, old_grid, vector_field, new_grid):
        #using RBF interpolation to interpolate the vector field from the old grid to the new grid
        x = np.linspace(np.min(old_grid[:, 0]), np.max(old_grid[:, 0]), 25)
        y = np.linspace(np.min(old_grid[:, 1]), np.max(old_grid[:, 1]), 10)
        z = np.linspace(np.min(old_grid[:, 2]), np.max(old_grid[:, 2]), 10)
        u = vector_field[:, 0].reshape(z.shape[0], y.shape[0], x.shape[0])
        v = vector_field[:, 1].reshape(z.shape[0], y.shape[0], x.shape[0])
        w = vector_field[:, 2].reshape(z.shape[0], y.shape[0], x.shape[0])
        swap_grid = np.zeros_like(new_grid)
        swap_grid[:, 0] = new_grid[:, 2]
        swap_grid[:, 1] = new_grid[:, 1]
        swap_grid[:, 2] = new_grid[:, 0]

        rgi_u = RegularGridInterpolator((z, y, x), u, method='quintic', bounds_error=True)
        rgi_v = RegularGridInterpolator((z, y, x), v, method='quintic', bounds_error=False)
        rgi_w = RegularGridInterpolator((z, y, x), w, method='quintic', bounds_error=False)
        u_new = rgi_u(swap_grid)
        v_new = rgi_v(swap_grid)
        w_new = rgi_w(swap_grid)

        return np.array([u_new, v_new, w_new]).T

        
    def compute_metrics(self):
        """
        Compute L2 error and MSE for each sample.
        """

        pred = self.MO_MapLR.position.value - self.MO_MapLR.rest_position.value
        gt = self.MO_MapHR.position.value - self.MO_MapHR.rest_position.value

        # Compute metrics only for non-zero displacements
        error = (gt - pred).reshape(-1)
        self.l2_error.append(np.linalg.norm(error))
        self.MSE_error.append((error.T @ error) / error.shape[0])
        self.l2_deformation.append(np.linalg.norm(gt))
        self.MSE_deformation.append((gt.reshape(-1).T @ gt.reshape(-1)) / gt.shape[0])

        self.RMSE_error.append(np.sqrt((error.T @ error) / error.shape[0]))
        self.RMSE_deformation.append(np.sqrt((gt.reshape(-1).T @ gt.reshape(-1)) / gt.shape[0] ))
    def compute_metrics_FC(self):
        """
        Compute L2 error and MSE for each sample for the FC model.
        """

        pred_FC = self.MO_MapLR_FC.position.value - self.MO_MapLR_FC.rest_position.value
        gt_FC = self.MO_MapHR.position.value - self.MO_MapHR.rest_position.value

        # Compute metrics only for non-zero displacements
        error_FC = (gt_FC - pred_FC).reshape(-1)
        self.l2_error_FC.append(np.linalg.norm(error_FC))
        self.MSE_error_FC.append((error_FC.T @ error_FC) / error_FC.shape[0])
        self.l2_deformation_FC.append(np.linalg.norm(gt_FC))
        self.MSE_deformation_FC.append((gt_FC.reshape(-1).T @ gt_FC.reshape(-1)) / gt_FC.shape[0])

        self.RMSE_error_FC.append(np.sqrt((error_FC.T @ error_FC) / error_FC.shape[0]))
        self.RMSE_deformation_FC.append(np.sqrt((gt_FC.reshape(-1).T @ gt_FC.reshape(-1)) / gt_FC.shape[0]))

        # #ADD Relative RMSE
        # self.RRMSE_error.append(np.sqrt(((error.T @ error) / error.shape[0]) / ((gt.reshape(-1).T @ gt.reshape(-1)))))

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

        if len(self.l2_error) > 0:
            print("\nGNN L2 ERROR Statistics :")
            print(f"\t- Distribution : {np.round(np.mean(self.l2_error), 6)} ± {np.round(np.std(self.l2_error), 6)} m")
            print(f"\t- Extrema : {np.round(np.min(self.l2_error), 6)} -> {np.round(np.max(self.l2_error), 6)} m")
            relative_error = np.array(self.l2_error) / np.array(self.l2_deformation)
            print(f"\t- Relative Distribution : {np.round(1e2 * relative_error.mean(), 6)} ± {np.round(1e2 * relative_error.std(), 6)} %")
            print(f"\t- Relative Extrema : {np.round(1e2 * relative_error.min(), 6)} -> {np.round(1e2 * relative_error.max(), 6)} %")

            print("\nGNN MSE Statistics :")
            print(f"\t- Distribution : {np.round(np.mean(self.MSE_error), 6)} ± {np.round(np.std(self.MSE_error), 6)} m²")
            print(f"\t- Extrema : {np.round(np.min(self.MSE_error), 6)} -> {np.round(np.max(self.MSE_error), 6)} m²")
            relative_error = np.array(self.MSE_error) / np.array(self.MSE_deformation)
            print(f"\t- Relative Distribution : {np.round(1e2 * relative_error.mean(), 6)} ± {np.round(1e2 * relative_error.std(), 6)} %")
            print(f"\t- Relative Extrema : {np.round(1e2 * relative_error.min(), 6)} -> {np.round(1e2 * relative_error.max(), 6)} %")

            print("\nGNN RMSE Statistics :")
            print(f"\t- Distribution : {np.round(np.mean(self.RMSE_error), 6)} ± {np.round(np.std(self.RMSE_error), 6)} m")
            print(f"\t- Extrema : {np.round(np.min(self.RMSE_error), 6)} -> {np.round(np.max(self.RMSE_error), 6)} m")
            relative_error = np.array(self.RMSE_error) / np.array(self.RMSE_deformation)
            print(f"\t- Relative Distribution : {np.round(1e2 * relative_error.mean(), 6)} ± {np.round(1e2 * relative_error.std(), 6)} %")
            print(f"\t- Relative Extrema : {np.round(1e2 * relative_error.min(), 6)} -> {np.round(1e2 * relative_error.max(), 6)} %")

            print("\nFC L2 ERROR Statistics :")
            print(f"\t- Distribution : {np.round(np.mean(self.l2_error_FC), 6)} ± {np.round(np.std(self.l2_error_FC), 6)} m")
            print(f"\t- Extrema : {np.round(np.min(self.l2_error_FC), 6)} -> {np.round(np.max(self.l2_error_FC), 6)} m")
            relative_error_FC = np.array(self.l2_error_FC) / np.array(self.l2_deformation_FC)
            print(f"\t- Relative Distribution : {np.round(1e2 * relative_error_FC.mean(), 6)} ± {np.round(1e2 * relative_error_FC.std(), 6)} %")
            print(f"\t- Relative Extrema : {np.round(1e2 * relative_error_FC.min(), 6)} -> {np.round(np.max(relative_error_FC), 6)} %")

            print("\nFC MSE Statistics :")
            print(f"\t- Distribution : {np.round(np.mean(self.MSE_error_FC), 6)} ± {np.round(np.std(self.MSE_error_FC), 6)} m²")
            print(f"\t- Extrema : {np.round(np.min(self.MSE_error_FC), 6)} -> {np.round(np.max(self.MSE_error_FC), 6)} m²")
            relative_error_FC = np.array(self.MSE_error_FC) / np.array(self.MSE_deformation_FC)
            print(f"\t- Relative Distribution : {np.round(1e2 * relative_error_FC.mean(), 6)} ± {np.round(1e2 * relative_error_FC.std(), 6)} %")
            print(f"\t- Relative Extrema : {np.round(1e2 * relative_error_FC.min(), 6)} -> {np.round(np.max(relative_error_FC), 6)} %")

            print("\nFC RMSE Statistics :")
            print(f"\t- Distribution : {np.round(np.mean(self.RMSE_error_FC), 6)} ± {np.round(np.std(self.RMSE_error_FC), 6)} m")
            print(f"\t- Extrema : {np.round(np.min(self.RMSE_error_FC), 6)} -> {np.round(np.max(self.RMSE_error_FC), 6)} m")
            relative_error_FC = np.array(self.RMSE_error_FC) / np.array(self.RMSE_deformation_FC)
            print(f"\t- Relative Distribution : {np.round(1e2 * relative_error_FC.mean(), 6)} ± {np.round(1e2 * relative_error_FC.std(), 6)} %")
            print(f"\t- Relative Extrema : {np.round(1e2 * relative_error_FC.min(), 6)} -> {np.round(np.max(relative_error_FC), 6)} %")

            #add a table comparing rmse of the two models
            print("\nComparison of the RMSE between the GNN and the FC model:")
            print(f"\t- GNN RMSE : {np.round(np.mean(self.RMSE_error), 6)} ± {np.round(np.std(self.RMSE_error), 6)} m")
            print(f"\t- FC RMSE : {np.round(np.mean(self.RMSE_error_FC), 6)} ± {np.round(np.std(self.RMSE_error_FC), 6)} m")

        elif len(self.errs) > 0:
            # plot the error as a function of the noise
            import matplotlib.pyplot as plt
            plt.semilogx(self.noises, self.errs)
            plt.xlabel('Noise')
            plt.ylabel('Error')
            plt.title('Error as a function of the noise')
            plt.show()
        else:
            print("No data to compute metrics.")

    
        

        

def createScene(rootNode, *args, **kwargs):
    rootNode.dt = 0.01
    rootNode.gravity = [0, 0, 0]
    rootNode.name = 'root'
    asc = AnimationStepController(rootNode, *args, **kwargs)
    rootNode.addObject(asc)
    return rootNode, asc


def main():
    import Sofa.Gui
    SofaRuntime.importPlugin("Sofa.GL.Component.Rendering3D")
    SofaRuntime.importPlugin("Sofa.GL.Component.Shader")
    SofaRuntime.importPlugin("Sofa.Component.StateContainer")
    SofaRuntime.importPlugin("Sofa.Component.ODESolver.Backward")
    SofaRuntime.importPlugin("Sofa.Component.LinearSolver.Direct")
    SofaRuntime.importPlugin("Sofa.Component.IO.Mesh")
    SofaRuntime.importPlugin("Sofa.Component.MechanicalLoad")
    SofaRuntime.importPlugin("Sofa.Component.Engine.Select")
    SofaRuntime.importPlugin("Sofa.Component.SolidMechanics.FEM.Elastic")

    root=Sofa.Core.Node("root")
    rootNode, asc = createScene(root)
    Sofa.Simulation.init(root)

    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(800, 600)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()
    asc.close()

if __name__ == '__main__':
    main()
