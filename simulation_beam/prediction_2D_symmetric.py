from httplib2 import ProxiesUnavailableError
from pyparsing import rest_of_line
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

from network.fully_connected_2D import Trainer as Trainer2D
from network.FC_Error_estimation import Trainer as Trainer
from parameters_2D import p_grid, p_grid_LR, p_grid_test

from scipy.interpolate import RBFInterpolator, griddata

import SofaCaribou
#SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_ROOT'])

np.random.seed(42)

class AnimationStepController(Sofa.Core.Controller):
    def __init__(self, node, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.externalForce = [0, -10, 0]
        self.createGraph(node)
        self.root = node
        self.save = False
        self.save_for_images = False
        self.l2_error, self.MSE_error = [], []
        self.l2_deformation, self.MSE_deformation = [], []
        self.RMSE_error, self.RMSE_deformation = [], []
        self.RRMSE_error, self.RRMSE_deformation = [], []
        self.network = Trainer('npy_beam/2024-08-12_17:11:01_symmetric/train', 32, 0.001, 500)
        # self.network.load_model('models/model_2024-05-22_10:25:12.pth') # efficient
        # self.network.load_model('models/model_2024-05-21_14:58:44.pth') # not efficient
        self.network.load_model('models/model_2024-08-14_15:07:58_symmetric_symlog.pth') # efficient noisy

    def createGraph(self, rootNode):

        rootNode.addObject('RequiredPlugin', name='MultiThreading')
        #rootNode.addObject('RequiredPlugin', name='SofaCaribou')
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
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.IO.Mesh') # Needed to use components [MeshGmshLoader]

        rootNode.addObject('DefaultAnimationLoop')
        rootNode.addObject('DefaultVisualManagerLoop')
        rootNode.addObject('VisualStyle', name="visualStyle", displayFlags="showBehaviorModels")# showCollisionModels")
        
        sphereRadius=0.025

        filename_high = 'mesh/beam_5080.msh'
        filename_low = 'mesh/beam_653.msh'
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
        self.exactSolution.addObject('BoxROI', name='ROI', box="-0.1 -1.1 -1.1 0.1 1.1 1.1")
        self.cff2 = self.exactSolution.addObject('ConstantForceField', indices="@ROI.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff_box = self.exactSolution.addObject('BoxROI', name='ROI2', box="9.9 -1.1 -1.1 10.1 1.1 1.1")
        self.cff = self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.exactSolution.addObject('BoxROI', name='ROI3', box="4.9 -1.1 -1.1 5.1 1.1 1.1")
        self.exactSolution.addObject('FixedConstraint', indices = '@ROI3.indices')

        self.mapping = self.exactSolution.addChild("SamplingMapping")
        self.MO_MapHR = self.mapping.addObject('MechanicalObject', name='DOFs_HR', template='Vec3d', src='@../../SamplingNodes/coarseGridHigh')
        #self.MO1_HR = self.mapping.addObject('MechanicalObject', name='DOFs_HR', template='Vec3d', position='1 3 0')
        self.mapping.addObject('BarycentricMapping', name="mapping", input='@../DOFs', input_topology='@../triangleTopo', output='@DOFs_HR')
        self.mapping.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 0')

        self.exactSolution.addChild("visual")
        self.exactSolution.visual.addObject('OglModel', src='@../grid', color='0 1 1 0.8')
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
        self.cff_box2_LR = self.LowResSolution.addObject('BoxROI', name='ROI', box='-0.1 -1.1 -1.1 0.1 1.1 1.1')
        self.cff2LR = self.LowResSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff_box_LR = self.LowResSolution.addObject('BoxROI', name='ROI2', box="9.9 -1.1 -1.1 10.1 1.1 1.1")
        self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.LowResSolution.addObject('BoxROI', name='ROI3', box="4.9 -1.1 -1.1 5.1 1.1 1.1")
        self.LowResSolution.addObject('FixedConstraint', indices='@ROI3.indices')

        self.mapping = self.LowResSolution.addChild("SamplingMapping")
        self.MO_MapLR = self.mapping.addObject('MechanicalObject', name='DOFs_HR', template='Vec3d', src='@../../SamplingNodes/coarseGridHigh')
        #self.MO1_LR = self.mapping.addObject('MechanicalObject', name='DOFs_HR', template='Vec3d', position='1 3 0')
        self.mapping.addObject('BarycentricMapping', name="mapping", input='@../DOFs', input_topology='@../triangleTopo', output='@DOFs_HR')
        self.mapping.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 1 0.9')


        # self.trained_nodes = rootNode.addChild('SparseCoarseMesh')
        # self.trained_nodes.addObject('SparseGridTopology', n="10 10 1", position='@../grid.position', name='coarseGridLow')
        # self.trained_nodes.addObject('TriangleSetTopologyContainer', name='triangleTopoLow', src='@coarseGridLow')
        # self.MO_training = self.trained_nodes.addObject('MechanicalObject', name='coarseDOFsLow', template='Vec3d', src='@coarseGridLow')
        # self.trained_nodes.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='1 1 0')
        # self.trained_nodes.addObject('BarycentricMapping', name="mapping", input='@DOFs', input_topology='@triangleTopo', output='@coarseDOFsLow', output_topology='@triangleTopoLow')

        self.LowResSolution.addChild("visual")
        self.visual_model = self.LowResSolution.visual.addObject('OglModel', src='@../gridLow', color='1 0 0')
        self.LowResSolution.visual.addObject('IdentityMapping', input='@../DOFs', output='@./')

        # self.LowResSolution.addChild("visual_noncorrected")
        # self.LowResSolution.visual_noncorrected.addObject('OglModel', src='@../../loader', color='0 1 0 0.5')
        # self.LowResSolution.visual_noncorrected.addObject('BarycentricMapping', input='@../DOFs', output='@./')


        print("High resolution shape: ", self.MO_sampling.position.value.shape)
        print("Low resolution shape: ", self.MO2.position.value.shape)
        self.high_res_shape = self.MO_MapHR.position.value.shape
        self.low_res_shape = self.MO_MapLR.position.value.shape


    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        """

        print("Simulation initialized.")
        print("High resolution shape: ", self.high_res_shape)
        print("Low resolution shape: ", self.low_res_shape)
        self.inputs = []
        self.outputs = []
        self.save = False
        self.efficient_sampling = False
        self.start_time = 0
        self.directory = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        if self.save:
            if not os.path.exists('npy'):
                os.mkdir('npy')
            # get current time from computer format yyyy-mm-dd-hh-mm-ss and create a folder with that name
            
            os.makedirs(f'npy/{self.directory}')
            print(f"Saving data to npy/{self.directory}")

        print("High resolution shape: ", self.MO_sampling.position.value.shape)
        print("Low resolution shape: ", self.MO_MapLR.position.value.shape)
        self.high_res_shape = self.MO_MapHR.position.value.shape
        self.low_res_shape = self.MO_MapLR.position.value.shape

        surface = self.surface_topo
        surface_LR = self.surface_topo_LR

        self.sampled = False

        self.idx_surface = surface.triangles.value.reshape(-1)
        self.idx_surface_LR = surface_LR.triangles.value.reshape(-1)

    def onAnimateBeginEvent(self, event):

        # reset positions
        self.MO1.position.value = self.MO1.rest_position.value
        self.MO2.position.value = self.MO2.rest_position.value
        if self.sampled:
            print("================== Sampled all magnitudes and versors ==================\n")
            print ("================== The simulation is over ==================\n")
        
        if not self.efficient_sampling:
            self.theta = np.random.uniform(0, 2*np.pi)
            self.z = np.random.uniform(-1, 1)
            self.versor_right = np.array([np.sqrt(1 - self.z**2)*np.cos(self.theta), np.sqrt(1 - self.z**2)*np.sin(self.theta), self.z])
            self.versor_left = np.array([-np.sqrt(1 - self.z**2)*np.cos(self.theta), np.sqrt(1 - self.z**2)*np.sin(self.theta), self.z])
            self.magnitude = np.random.uniform(0, 200)
            self.externalForce_right = self.magnitude * self.versor_right
            self.externalForce_left = self.magnitude * self.versor_left
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

        self.exactSolution.removeObject(self.cff2)
        self.exactSolution.removeObject(self.cff)
        self.LowResSolution.removeObject(self.cff2LR)
        self.LowResSolution.removeObject(self.cffLR)

        self.cff2 = self.exactSolution.addObject('ConstantForceField', indices="@ROI.indices", totalForce=self.externalForce_left, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff = self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce_right, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff2LR = self.LowResSolution.addObject('ConstantForceField', indices="@ROI.indices", totalForce=self.externalForce_left, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce_right, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff.init()
        self.cff2.init()
        self.cffLR.init()
        self.cff2LR.init()
        self.start_time = process_time()


    def onAnimateEndEvent(self, event):
        
        # compute the displacement between the high and low resolution solutions
       



        coarse_pos = self.MO_MapLR.position.value.copy() - self.MO_MapLR.rest_position.value.copy()
        
        # print("Coarse position: ", coarse_pos.shape)
        # cut the z component
        # coarse_pos = coarse_pos[:, :2]
        # print("Coarse position shape: ", coarse_pos.shape)
        inputs = np.reshape(coarse_pos, -1)


        # self.noises = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
        # self.errs = []
        # for i in self.noises:
        #     print(f"Adding noise: {i}")
        #     self.MO_training.position.value = coarse_pos + self.MO_training.rest_position.value
        #     # add noise to the input
        #     noise = np.random.normal(0, i, inputs.shape)
        #     noisy_inputs = inputs + noise


        U = self.network.predict(inputs).cpu().numpy()

        # print("Predicted displacement: ", U.shape)
        # print("Low res shape: ", self.low_res_shape)
        # reshape U to have the same shape as the position
        # add the z component

        

        U = np.reshape(U, (self.low_res_shape[0], self.low_res_shape[1]))
        # U = np.append(U, np.zeros((self.high_res_shape[0], 1)), axis=1)
        # print("U shape after reshape: ", U)
        # compute L2 norm of the prediction error

        # print("Predicted displacement first 5 nodes: ", U[:5])
        # print("Exact displacement first 5 nodes: ", self.MO1.position.value[:5] - self.MO1.rest_position.value[:5])


        self.MO_MapLR.position.value = self.MO_MapLR.position.value.copy() + U

        

            # err = np.linalg.norm(self.MO1_LR.position.value - self.MO_training.position.value)
            # print(f"Prediction error: {err}")
            # self.errs.append(err)
                
        positions = self.MO_MapLR.rest_position.value.copy()
        #print("Positions: ", positions)
        #print("Rest position shape: ", self.MO_training.position.value)
        displacement = self.MO_MapLR.position.value.copy() - self.MO_MapLR.rest_position.value.copy()
        #print("Displacement: ", displacement)

        interpolator = RBFInterpolator(positions, displacement, neighbors=5, kernel="linear")
        interpolate_positions = self.MO2.rest_position.value.copy()
        corrected_displacement = interpolator(interpolate_positions)

        # print("Corrected displacement: ", corrected_displacement)
        #print("Before correction: ", self.MO2.position.value)
        
        self.MO2.position.value = interpolate_positions + corrected_displacement
        self.visual_model.position.value = interpolate_positions + corrected_displacement
        # #print("After correction: ", self.MO2.position.value)

        # ============== UPDATE THE NN MODEL ==============
        # self.MO_NN.position.value = self.MO2.position.value + U

        self.end_time = process_time()
        pred = self.MO_MapLR.position.value.copy() - self.MO_MapLR.rest_position.value.copy()
        gt = self.MO_MapHR.position.value.copy() - self.MO_MapHR.rest_position.value.copy()
        self.compute_metrics(pred, gt)
        print("Computation time for 1 time step: ", self.end_time - self.start_time)
        print("External force: ", np.linalg.norm(self.externalForce))
        
        
        error = (pred - gt).reshape(-1)
        print("L2 error divided by number of points: ", np.linalg.norm(error)**2 / error.shape[0])
        print("MSE error: ", (error.T @ error) / error.shape[0])


        if self.save_for_images:
            if not os.path.exists('images_data'):
                os.mkdir('images_data')

            if not os.path.exists(f'images_data/{self.directory}'):
                os.mkdir(f'images_data/{self.directory}')

            ground_truth_displacement = self.MO1.position.value - self.MO1.rest_position.value
            prediction_displacement = self.MO2.position.value - self.MO2.rest_position.value

            ground_truth_grid = self.MO_MapHR.position.value - self.MO_MapHR.rest_position.value
            prediction_grid = self.MO_MapLR.position.value - self.MO_MapLR.rest_position.value

            ground_truth_rest = self.MO1.rest_position.value
            prediction_rest = self.MO2.rest_position.value

            ground_truth_grid_rest = self.MO_MapHR.rest_position.value
            prediction_grid_rest = self.MO_MapLR.rest_position.value

            np.save(f'images_data/{self.directory}/ground_truth_displacement.npy', ground_truth_displacement)
            np.save(f'images_data/{self.directory}/prediction_displacement.npy', prediction_displacement)
            np.save(f'images_data/{self.directory}/ground_truth_grid.npy', ground_truth_grid)
            np.save(f'images_data/{self.directory}/prediction_grid.npy', prediction_grid)
            np.save(f'images_data/{self.directory}/ground_truth_rest.npy', ground_truth_rest)
            np.save(f'images_data/{self.directory}/prediction_rest.npy', prediction_rest)
            np.save(f'images_data/{self.directory}/ground_truth_grid_rest.npy', ground_truth_grid_rest)
            np.save(f'images_data/{self.directory}/prediction_grid_rest.npy', prediction_grid_rest)
   


    def compute_metrics(self, pred, gt):
        """
        Compute L2 error and MSE for each sample.
        """

        # pred = self.MO_MapLR.position.value - self.MO_MapLR.rest_position.value
        # gt = self.MO_MapHR.position.value - self.MO_MapHR.rest_position.value

        # Compute metrics only for non-small displacements
        if np.linalg.norm(gt) > 1e-6:
            error = (gt - pred).reshape(-1)
            self.l2_error.append(np.linalg.norm(error))
            self.MSE_error.append((error.T @ error) / error.shape[0])
            self.l2_deformation.append(np.linalg.norm(gt))
            self.MSE_deformation.append((gt.reshape(-1).T @ gt.reshape(-1)) / gt.shape[0])
            self.RMSE_error.append(np.sqrt((error.T @ error) / error.shape[0]))
            self.RMSE_deformation.append(np.sqrt((gt.reshape(-1).T @ gt.reshape(-1)) / gt.shape[0] ))

            #ADD Relative RMSE
            self.RRMSE_error.append(np.sqrt(((error.T @ error) / error.shape[0]) / ((gt.reshape(-1).T @ gt.reshape(-1)))))

    def close(self):

        if len(self.l2_error) > 0:
            print("\nL2 ERROR Statistics :")
            print(f"\t- Distribution : {np.round(np.mean(self.l2_error), 6)} ± {np.round(np.std(self.l2_error), 6)} m")
            print(f"\t- Extrema : {np.round(np.min(self.l2_error), 6)} -> {np.round(np.max(self.l2_error), 6)} m")
            relative_error = np.array(self.l2_error) / np.array(self.l2_deformation)
            print(f"\t- Relative Distribution : {np.round(1e2 * relative_error.mean(), 6)} ± {np.round(1e2 * relative_error.std(), 6)} %")
            print(f"\t- Relative Extrema : {np.round(1e2 * relative_error.min(), 6)} -> {np.round(1e2 * relative_error.max(), 6)} %")

            print("\nMSE Statistics :")
            print(f"\t- Distribution : {np.round(np.mean(self.MSE_error), 6)} ± {np.round(np.std(self.MSE_error), 6)} m²")
            print(f"\t- Extrema : {np.round(np.min(self.MSE_error), 6)} -> {np.round(np.max(self.MSE_error), 6)} m²")
            relative_error = np.array(self.MSE_error) / np.array(self.MSE_deformation)
            print(f"\t- Relative Distribution : {np.round(1e2 * relative_error.mean(), 6)} ± {np.round(1e2 * relative_error.std(), 6)} %")
            print(f"\t- Relative Extrema : {np.round(1e2 * relative_error.min(), 6)} -> {np.round(1e2 * relative_error.max(), 6)} %")

            print("\nRMSE Statistics :")
            print(f"\t- Distribution : {np.round(np.mean(self.RMSE_error), 6)} ± {np.round(np.std(self.RMSE_error), 6)} m")
            print(f"\t- Extrema : {np.round(np.min(self.RMSE_error), 6)} -> {np.round(np.max(self.RMSE_error), 6)} m")
            relative_error = np.array(self.RMSE_error) / np.array(self.RMSE_deformation)
            print(f"\t- Relative Distribution : {np.round(1e2 * relative_error.mean(), 6)} ± {np.round(1e2 * relative_error.std(), 6)} %")
            print(f"\t- Relative Extrema : {np.round(1e2 * relative_error.min(), 6)} -> {np.round(1e2 * relative_error.max(), 6)} %")

            print("\nRRMSE Statistics :")
            print(f"\t- Distribution : {np.round(np.mean(self.RRMSE_error), 6)} ± {np.round(np.std(self.RRMSE_error), 6)} m")
            print(f"\t- Extrema : {np.round(np.min(self.RRMSE_error), 6)} -> {np.round(np.max(self.RRMSE_error), 6)} m")

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
