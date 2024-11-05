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

sys.path.append(os.path.join(os.path.dirname(__file__), '../network'))

print(sys.path)

from network.GNN_Error_estimation import Trainer as Trainer
from network.FC_Error_estimation_integrated import Trainer as TrainerFC
from parameters_2D import p_grid, p_grid_LR
from torch_geometric.data import Data

from scipy.interpolate import RBFInterpolator, griddata



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

        self.network = Trainer('npy_GNN/2024-11-03_18:32:29_estimation', 16, 0.001, 500)
        self.network.load_model('models_GNN/model_2024-11-03_23:02:55_GNN.pth')

        self.networkFC = TrainerFC('npy_GNN/2024-11-03_18:32:29_estimation', 16, 0.001,  500)
        self.networkFC.load_model('models_FC/model_2024-11-04_18:52:44_FC.pth')
        
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


        self.exactSolution = rootNode.addChild('HighResSolution2D', activated=True)
        self.exactSolution.addObject('MeshGmshLoader', name='grid', filename='mesh/rectangle_1166.msh')
        self.surface_topo = self.exactSolution.addObject('TriangleSetTopologyContainer', name='triangleTopo', src='@grid')
        self.MO1 = self.exactSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        # self.exactSolution.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@quadTopo")
        self.exactSolution.addObject('StaticSolver', name='ODE', newton_iterations="20", printLog=True)
        self.exactSolution.addObject('CGLinearSolver', iterations=1000, name="linear solver", tolerance="1.0e-6", threshold="1.0e-6") 
        self.exactSolution.addObject('TriangularFEMForceField', name="FEM", youngModulus=5000, poissonRatio=0.4, method="large")
        self.exactSolution.addObject('BoxROI', name='ROI', box=p_grid.fixed_box)
        self.exactSolution.addObject('FixedConstraint', indices='@ROI.indices')
        self.cff_box = self.exactSolution.addObject('BoxROI', name='ROI2', box="9.9 -1.1 -0.1 10.1 1.1 0.1")
        self.cff = self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.coarse = self.exactSolution.addChild('CoarseMesh')
        self.coarse.addObject('RegularGridTopology', name='coarseGrid', min=p_grid_LR.min, max=p_grid_LR.max, nx=p_grid_LR.res[0], ny=p_grid_LR.res[1], nz=p_grid_LR.res[2])
        self.coarse.addObject('TriangleSetTopologyContainer', name='triangleTopo', src='@coarseGrid')
        self.MO1_LR = self.coarse.addObject('MechanicalObject', name='coarseDOFs', template='Vec3d', src='@coarseGrid')
        self.coarse.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 0')
        self.coarse.addObject('BarycentricMapping', name="mapping", input='@../DOFs', input_topology='@../triangleTopo', output='@coarseDOFs', output_topology='@triangleTopo')


        self.exactSolution.addChild("visual")
        self.exactSolution.visual.addObject('OglModel', src='@../grid', color='0 1 1 0.5')
        self.exactSolution.visual.addObject('IdentityMapping', input='@../DOFs', output='@./')

        # same object with different resolution

        self.LowResSolution = rootNode.addChild('LowResSolution2D', activated=True)
        self.LowResSolution.addObject('MeshGmshLoader', name='grid', filename='mesh/rectangle_75.msh')
        self.surface_topo_LR = self.LowResSolution.addObject('TriangleSetTopologyContainer', name='quadTopo', src='@grid')
        self.MO2 = self.LowResSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        # self.LowResSolution.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@quadTopo")
        self.LowResSolution.addObject('StaticSolver', name='ODE', newton_iterations="20", printLog=True)
        self.LowResSolution.addObject('CGLinearSolver', iterations=500, name="linear solver", tolerance="1.0e-6", threshold="1.0e-6")
        self.LowResSolution.addObject('TriangularFEMForceField', name="FEM", youngModulus=5000, poissonRatio=0.4, method="large")
        self.LowResSolution.addObject('BoxROI', name='ROI', box=p_grid_LR.fixed_box)
        self.LowResSolution.addObject('FixedConstraint', indices='@ROI.indices')
        self.cff_box_LR = self.LowResSolution.addObject('BoxROI', name='ROI2', box="9.9 -1.1 -0.1 10.1 1.1 0.1")
        self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")

        self.trained_nodes = self.LowResSolution.addChild('CoarseMesh')
        self.trained_nodes.addObject('RegularGridTopology', name='coarseGrid', min=p_grid_LR.min, max=p_grid_LR.max, nx=p_grid_LR.res[0], ny=p_grid_LR.res[1], nz=p_grid_LR.res[2])
        self.trained_nodes.addObject('TriangleSetTopologyContainer', name='triangleTopo', src='@coarseGrid')
        self.MO_training = self.trained_nodes.addObject('MechanicalObject', name='coarseDOFs', template='Vec3d', src='@coarseGrid')
        self.trained_nodes.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 0')
        self.trained_nodes.addObject('BarycentricMapping', name="mapping", input='@../DOFs', input_topology='@../triangleTopo', output='@coarseDOFs', output_topology='@triangleTopo')

        self.LowResSolution.addChild("visual")
        self.visual_model = self.LowResSolution.visual.addObject('OglModel', src='@../grid', color='1 0 0 0.2')
        self.LowResSolution.visual.addObject('IdentityMapping', input='@../DOFs', output='@./')

        # fc model

        self.LowResSolution_FC = rootNode.addChild('LowResSolution2D_FC', activated=True)
        self.LowResSolution_FC.addObject('MeshGmshLoader', name='grid', filename='mesh/rectangle_75.msh')
        self.surface_topo_LR_FC = self.LowResSolution_FC.addObject('TriangleSetTopologyContainer', name='quadTopo', src='@grid')
        self.MO2_FC = self.LowResSolution_FC.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        # self.LowResSolution_FC.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@quadTopo")
        self.LowResSolution_FC.addObject('StaticSolver', name='ODE', newton_iterations="20", printLog=True)
        self.LowResSolution_FC.addObject('CGLinearSolver', iterations=500, name="linear solver", tolerance="1.0e-6", threshold="1.0e-6")
        self.LowResSolution_FC.addObject('TriangularFEMForceField', name="FEM", youngModulus=5000, poissonRatio=0.4, method="large")
        self.LowResSolution_FC.addObject('BoxROI', name='ROI', box=p_grid_LR.fixed_box)
        self.LowResSolution_FC.addObject('FixedConstraint', indices='@ROI.indices')
        self.cff_box_LR_FC = self.LowResSolution_FC.addObject('BoxROI', name='ROI2', box="9.9 -1.1 -0.1 10.1 1.1 0.1")
        self.cffLR_FC = self.LowResSolution_FC.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")

        self.trained_nodes_FC = self.LowResSolution_FC.addChild('CoarseMesh')
        self.trained_nodes_FC.addObject('RegularGridTopology', name='coarseGrid', min=p_grid_LR.min, max=p_grid_LR.max, nx=p_grid_LR.res[0], ny=p_grid_LR.res[1], nz=p_grid_LR.res[2])
        self.trained_nodes_FC.addObject('TriangleSetTopologyContainer', name='triangleTopo', src='@coarseGrid')
        self.MO_training_FC = self.trained_nodes_FC.addObject('MechanicalObject', name='coarseDOFs', template='Vec3d', src='@coarseGrid')
        self.trained_nodes_FC.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 0')
        self.trained_nodes_FC.addObject('BarycentricMapping', name="mapping", input='@../DOFs', input_topology='@../triangleTopo', output='@coarseDOFs', output_topology='@triangleTopo')

        self.LowResSolution.addChild("visual_FC")
        self.visual_model_FC = self.LowResSolution.visual_FC.addObject('OglModel', src='@../grid', color='1 0 0 0.2')
        self.LowResSolution.visual_FC.addObject('IdentityMapping', input='@../DOFs', output='@./')

        

        self.low_res_shape = (p_grid_LR.res[0]*p_grid_LR.res[1], 3)
        self.high_res_shape = (p_grid.res[0]*p_grid.res[1], 3)

        self.nb_nodes = len(self.MO1.position.value)
        self.nb_nodes_LR = len(self.MO1_LR.position.value)


    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        """

        print("Simulation initialized.")
        self.inputs = []
        self.outputs = []
        self.save = False
        self.start_time = 0
        self.directory = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        if self.save:
            if not os.path.exists('npy'):
                os.mkdir('npy')
            # get current time from computer format yyyy-mm-dd-hh-mm-ss and create a folder with that name
            os.makedirs(f'npy/{self.directory}')
            print(f"Saving data to npy/{self.directory}")

        surface = self.surface_topo
        surface_LR = self.surface_topo_LR
        surface_LR_Falserface_LR = surface_LR.triangles.value.reshape(-1)
        self.idx_surface_LR_FC = surface_LR_FC.triangles.value.reshape(-1)



    def onAnimateBeginEvent(self, event):
        self.bad_sample = False
        # reset positions
        self.MO1.position.value = self.MO1.rest_position.value
        self.MO2.position.value = self.MO2.rest_position.value
        self.MO2_FC.position.value = self.MO2_FC.rest_position.value
        # self.MO_NN.position.value = self.MO_NN.rest_position.value
        
        self.theta = np.random.uniform(0, 2*np.pi)
        self.versor = np.array([np.cos(self.theta), np.sin(self.theta)])
        self.magnitude = np.random.uniform(0, 70)
        self.externalForce = np.append(self.magnitude * self.versor, 0)

        #self.externalForce = [0, -60, 0]
        # self.externalForce_LR = [0, -60, 0]

        # Define random box
        side = np.random.randint(1, 4)
        if side == 2:
            x_min = 9.99
            x_max = 10.01
            y_min = np.random.uniform(-1.01, 0.0)
            y_max = y_min + 1
        elif side == 3:
            y_min = -1.01
            y_max = -0.99
            x_min = np.random.uniform(2.0, 9.0)
            x_max = x_min + 1
        else:
            x_min = np.random.uniform(2.0, 9.0)
            x_max = x_min + 1
            y_min = 0.99
            y_max = 1.01
            
        # Set the new bounding box
        self.exactSolution.removeObject(self.cff_box)
        self.cff_box = self.exactSolution.addObject('BoxROI', name='ForceBox', drawBoxes=False, drawSize=1,
                                            box=[x_min, y_min, -0.1, x_max, y_max, 0.1])
        self.cff_box.init()

        self.LowResSolution.removeObject(self.cff_box_LR)
        self.cff_box_LR = self.LowResSolution.addObject('BoxROI', name='ForceBox', drawBoxes=True, drawSize=1,
                                            box=[x_min, y_min, -0.1, x_max, y_max, 0.1])
        self.cff_box_LR.init()
        self.LowResSolution_FC.removeObject(self.cff_box_LR_FC)
        self.cff_box_LR_FC = self.LowResSolution_FC.addObject('BoxROI', name='ForceBox', drawBoxes=True, drawSize=1,
                                            box=[x_min, y_min, -0.1, x_max, y_max, 0.1])
        self.cff_box_LR_FC.init()

        # Get the intersection with the surface
        indices = list(self.cff_box.indices.value)
        indices = list(set(indices).intersection(set(self.idx_surface)))
        indices_LR = list(self.cff_box_LR.indices.value)
        indices_LR = list(set(indices_LR).intersection(set(self.idx_surface_LR)))
        indices_LR_FC = list(self.cff_box_LR_FC.indices.value)
        indices_LR_FC = list(set(indices_LR_FC).intersection(set(self.idx_surface_LR_FC)))
        self.exactSolution.removeObject(self.cff)
        self.cff = self.exactSolution.addObject('ConstantForceField', indices=indices, totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff.init()


        self.LowResSolution.removeObject(self.cffLR)
        self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices=indices_LR, totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cffLR.init()

        self.LowResSolution_FC.removeObject(self.cffLR_FC)
        self.cffLR_FC = self.LowResSolution_FC.addObject('ConstantForceField', indices=indices_LR_FC, totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cffLR_FC.init()

        if indices_LR == [] or indices == [] or indices_LR_FC == []:
            print("Empty intersection")
            self.bad_sample = True
        self.start_time = process_time()



    def onAnimateEndEvent(self, event):
        

        self.end_time = process_time()
        print("Computation time for 1 time step: ", self.end_time - self.start_time)
        print("External force: ", np.linalg.norm(self.externalForce))
        U_high = self.compute_displacement(self.MO1_LR)
        U_test = self.compute_displacement(self.MO1)
        U_low = self.compute_displacement(self.MO_training)
        U_low_FC = self.compute_displacement(self.MO_training_FC)
        edges_low = self.compute_edges(self.surface_topo_LR)
        print("U_low_FC: ", U_low_FC.shape)


        node_features = U_low
        edge_index = edges_low[:, :2].T
        edge_attr = edges_low[:, 2]
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

        data_FC = np.reshape(U_low_FC, (self.low_res_shape[0]*self.low_res_shape[1], 1))
        data_FC = th.tensor(data_FC, dtype=th.float32).T
        U = self.network.predict(data).cpu().numpy()
        U_FC = self.networkFC.predict(data_FC).cpu().numpy()
  

        

        U = np.reshape(U, (self.low_res_shape[0], self.low_res_shape[1]))
        U_FC = np.reshape(U_FC, (self.low_res_shape[0], self.low_res_shape[1]))


        self.MO_training.position.value = self.MO_training.position.value + U

        self.MO_training_FC.position.value = self.MO_training_FC.position.value + U_FC

                
        positions = self.MO_training.rest_position.value.copy()[:, :2]
        #print("Positions: ", positions)
        #print("Rest position shape: ", self.MO_training.position.value)
        displacement = self.MO_training.position.value.copy() - self.MO_training.rest_position.value.copy()
        displacement = displacement[:, :2]
        #print("Displacement: ", displacement)

        interpolator = RBFInterpolator(positions, displacement, neighbors=10, kernel="thin_plate_spline")
        interpolate_positions = self.MO2.rest_position.value.copy()
        interpolate_positions_2D = self.MO2.rest_position.value.copy()[:, :2]
        corrected_displacement = interpolator(interpolate_positions_2D)

        #print("Corrected displacement: ", corrected_displacement)
        #print("Before correction: ", self.MO2.position.value)
        corrected_displacement = np.append(corrected_displacement, np.zeros((interpolate_positions.shape[0], 1)), axis=1)
        self.MO2.position.value = interpolate_positions + corrected_displacement

        self.visual_model.position.value = interpolate_positions + corrected_displacement

        positions_FC = self.MO_training_FC.rest_position.value.copy()[:, :2]
        displacement_FC = self.MO_training_FC.position.value.copy() - self.MO_training_FC.rest_position.value.copy()
        displacement_FC = displacement_FC[:, :2]

        interpolator_FC = RBFInterpolator(positions_FC, displacement_FC, neighbors=10, kernel="thin_plate_spline")
        interpolate_positions_FC = self.MO2_FC.rest_position.value.copy()
        interpolate_positions_2D_FC = self.MO2_FC.rest_position.value.copy()[:, :2]
        corrected_displacement_FC = interpolator_FC(interpolate_positions_2D_FC)

        corrected_displacement_FC = np.append(corrected_displacement_FC, np.zeros((interpolate_positions_FC.shape[0], 1)), axis=1)
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
        print("L2 error: ", self.l2_error[-1])
        print("L2 deformation: ", self.l2_deformation[-1])
        print("Relative error: ", self.l2_error[-1]/np.linalg.norm(self.MO_training.position.value - self.MO_training.rest_position.value))

    def compute_metrics(self):
        """
        Compute L2 error and MSE for each sample.
        """

        pred = self.MO_training.position.value - self.MO_training.rest_position.value
        gt = self.MO1_LR.position.value - self.MO1_LR.rest_position.value

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

        pred_FC = self.MO_training_FC.position.value - self.MO_training_FC.rest_position.value
        gt_FC = self.MO1_LR.position.value - self.MO1_LR.rest_position.value

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
