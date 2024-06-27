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
SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_ROOT'])

class AnimationStepController(Sofa.Core.Controller):
    def __init__(self, node, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.externalForce = [0, -10, 0]
        self.createGraph(node)
        self.root = node
        self.save = False
        self.l2_error, self.MSE_error = [], []
        self.l2_deformation, self.MSE_deformation = [], []
        self.network = Trainer('npy_liver/2024-06-26_16:59:46_estimation_FHD/train', 32, 0.001, 1000)
        # self.network.load_model('models/model_2024-05-22_10:25:12.pth') # efficient
        # self.network.load_model('models/model_2024-05-21_14:58:44.pth') # not efficient
        self.network.load_model('models/model_2024-06-27_09:20:51_FHD.pth') # efficient noisy

    def createGraph(self, rootNode):

        rootNode.addObject('RequiredPlugin', name='MultiThreading')
        rootNode.addObject('RequiredPlugin', name='SofaCaribou')
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

        filename_high = 'mesh/liver_29494_smooth.msh'
        filename_low = 'mesh/liver_8879_smooth.msh'

        self.coarse = rootNode.addChild('SamplingNodes')
        self.coarse.addObject('MeshGmshLoader', name='grid', filename=filename_high, scale3d="1 1 1", translation="0 0 0")
        self.coarse.addObject('SparseGridTopology', n="50 50 50", position='@grid.position', name='coarseGridHigh')  # 

        self.coarse.addObject('TetrahedronSetTopologyContainer', name='triangleTopoHigh', src='@coarseGridHigh')
        self.MO_sampling = self.coarse.addObject('MechanicalObject', name='coarseDOFsHigh', template='Vec3d', src='@coarseGridHigh')
        self.coarse.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='1 0 0')
        #self.coarse.addObject('BarycentricMapping', name="mapping", input='@DOFs', input_topology='@triangleTopo', output='@coarseDOFsHigh', output_topology='@triangleTopoHigh')


        self.exactSolution = rootNode.addChild('HighResSolution2D', activated=True)
        self.exactSolution.addObject('MeshGmshLoader', name='grid', filename=filename_high)
        self.exactSolution.addObject('TetrahedronSetTopologyContainer', name='triangleTopo', src='@grid')
        self.MO1 = self.exactSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        # self.exactSolution.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@quadTopo")
        self.exactSolution.addObject('StaticSolver', name='ODE', newton_iterations="20", printLog=True)
        self.exactSolution.addObject('ParallelCGLinearSolver', template="ParallelCompressedRowSparseMatrixMat3x3d", iterations=20000, tolerance=1e-08, threshold=1e-08, warmStart=True)
        self.exactSolution.addObject('ParallelTetrahedronFEMForceField', name="FEM", youngModulus=5000, poissonRatio=0.4, method="large", updateStiffnessMatrix="false")
        self.exactSolution.addObject('BoxROI', name='ROI', box="-2.3 3.2 -0.3 -1.2 2.9 0.8", drawBoxes=True)
        self.exactSolution.addObject('FixedConstraint', indices="@ROI.indices")
        self.exactSolution.addObject('BoxROI', name='ROI2', box="-4.1 3.9 -0.1 -2.9 5.1 0.6", drawBoxes=True)
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
        self.LowResSolution.addObject('TetrahedronSetTopologyContainer', name='quadTopo', src='@gridLow')
        self.MO2 = self.LowResSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@gridLow')
        # self.LowResSolution.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@quadTopo")
        self.LowResSolution.addObject('StaticSolver', name='ODE', newton_iterations="10", printLog=True)
        self.LowResSolution.addObject('ParallelCGLinearSolver', template="ParallelCompressedRowSparseMatrixMat3x3d", iterations=8000, tolerance=1e-08, threshold=1e-08, warmStart=True) 
        self.LowResSolution.addObject('ParallelTetrahedronFEMForceField', name="FEM", youngModulus=5000, poissonRatio=0.4, method="large", updateStiffnessMatrix="false")
        self.LowResSolution.addObject('BoxROI', name='ROI', box="-2.3 3.2 -0.3 -1.2 2.9 0.8", drawBoxes=True)
        self.LowResSolution.addObject('FixedConstraint', indices="@ROI.indices")
        self.LowResSolution.addObject('BoxROI', name='ROI2', box="-4.1 3.9 -0.1 -2.9 5.1 0.6", drawBoxes=True)
        self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")

        self.mapping = self.LowResSolution.addChild("SamplingMapping")
        self.MO_MapLR = self.mapping.addObject('MechanicalObject', name='DOFs_HR', template='Vec3d', src='@../../SamplingNodes/coarseGridHigh')
        #self.MO1_LR = self.mapping.addObject('MechanicalObject', name='DOFs_HR', template='Vec3d', position='1 3 0')
        self.mapping.addObject('BarycentricMapping', name="mapping", input='@../DOFs', input_topology='@../triangleTopo', output='@DOFs_HR')
        self.mapping.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 1')


        # self.trained_nodes = rootNode.addChild('SparseCoarseMesh')
        # self.trained_nodes.addObject('SparseGridTopology', n="10 10 1", position='@../grid.position', name='coarseGridLow')
        # self.trained_nodes.addObject('TriangleSetTopologyContainer', name='triangleTopoLow', src='@coarseGridLow')
        # self.MO_training = self.trained_nodes.addObject('MechanicalObject', name='coarseDOFsLow', template='Vec3d', src='@coarseGridLow')
        # self.trained_nodes.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='1 1 0')
        # self.trained_nodes.addObject('BarycentricMapping', name="mapping", input='@DOFs', input_topology='@triangleTopo', output='@coarseDOFsLow', output_topology='@triangleTopoLow')

        self.LowResSolution.addChild("visual")
        self.LowResSolution.visual.addObject('OglModel', src='@../gridLow', color='1 0 0 0.2')
        self.LowResSolution.visual.addObject('IdentityMapping', input='@../DOFs', output='@./')


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
        self.start_time = 0
        if self.save:
            if not os.path.exists('npy'):
                os.mkdir('npy')
            # get current time from computer format yyyy-mm-dd-hh-mm-ss and create a folder with that name
            self.directory = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            os.makedirs(f'npy/{self.directory}')
            print(f"Saving data to npy/{self.directory}")

        print("High resolution shape: ", self.MO_sampling.position.value.shape)
        print("Low resolution shape: ", self.MO_MapLR.position.value.shape)
        self.high_res_shape = self.MO_MapHR.position.value.shape
        self.low_res_shape = self.MO_MapLR.position.value.shape

    def onAnimateBeginEvent(self, event):

        # reset positions
        self.MO1.position.value = self.MO1.rest_position.value
        self.MO2.position.value = self.MO2.rest_position.value
        # self.MO_NN.position.value = self.MO_NN.rest_position.value
        
        self.vector = np.random.uniform(-1, 1, 2)
        self.versor = self.vector / np.linalg.norm(self.vector)
        self.magnitude = np.random.uniform(50, 80)
        self.externalForce = np.append(self.magnitude * self.versor, 0)

        self.externalForce = [0, -60, 0]
        # self.externalForce_LR = [0, -60, 0]

        self.exactSolution.removeObject(self.cff)
        self.cff = self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.LowResSolution.removeObject(self.cffLR)
        self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff.init()
        self.cffLR.init()


        self.start_time = process_time()



    def onAnimateEndEvent(self, event):
        
        # compute the displacement between the high and low resolution solutions
       



        coarse_pos = self.MO_MapLR.position.value.copy() - self.MO_MapLR.rest_position.value.copy()
        
        print("Coarse position: ", coarse_pos.shape)
        # cut the z component
        # coarse_pos = coarse_pos[:, :2]
        # print("Coarse position shape: ", coarse_pos.shape)
        inputs = np.reshape(coarse_pos, -1)
        if self.network.normalized:
            scaler = MinMaxScaler()
            inputs = scaler.fit_transform(inputs)

        # self.noises = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
        # self.errs = []
        # for i in self.noises:
        #     print(f"Adding noise: {i}")
        #     self.MO_training.position.value = coarse_pos + self.MO_training.rest_position.value
        #     # add noise to the input
        #     noise = np.random.normal(0, i, inputs.shape)
        #     noisy_inputs = inputs + noise


        U = self.network.predict(inputs).cpu().numpy()
        if self.network.normalized:
            U = scaler.inverse_transform(U)
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


        self.MO_MapLR.position.value = self.MO_MapLR.position.value + U

        

            # err = np.linalg.norm(self.MO1_LR.position.value - self.MO_training.position.value)
            # print(f"Prediction error: {err}")
            # self.errs.append(err)
                
        # positions = self.MO_MapLR.rest_position.value.copy()[:, :2]
        # #print("Positions: ", positions)
        # #print("Rest position shape: ", self.MO_training.position.value)
        # displacement = self.MO_MapLR.position.value.copy() - self.MO_MapLR.rest_position.value.copy()
        # displacement = displacement[:, :2]
        # print("Displacement: ", displacement)

        # interpolator = RBFInterpolator(positions, displacement, neighbors=5, kernel="thin_plate_spline")
        # interpolate_positions = self.MO2.rest_position.value.copy()
        # interpolate_positions_2D = self.MO2.rest_position.value.copy()[:, :2]
        # corrected_displacement = interpolator(interpolate_positions_2D)

        # print("Corrected displacement: ", corrected_displacement)
        # #print("Before correction: ", self.MO2.position.value)
        # corrected_displacement = np.append(corrected_displacement, np.zeros((interpolate_positions.shape[0], 1)), axis=1)
        # self.MO2.position.value = interpolate_positions + corrected_displacement
        # #print("After correction: ", self.MO2.position.value)

        # ============== UPDATE THE NN MODEL ==============
        # self.MO_NN.position.value = self.MO2.position.value + U

        self.end_time = process_time()
        # error = np.linalg.norm(self.MO_NN.position.value - self.MO1.position.value)
        # print(f"Prediction error: {error}")
        self.compute_metrics()
        print("Computation time for 1 time step: ", self.end_time - self.start_time)
        print("External force: ", np.linalg.norm(self.externalForce))



   


    def compute_metrics(self):
        """
        Compute L2 error and MSE for each sample.
        """

        pred = self.MO_MapLR.position.value - self.MO_MapLR.rest_position.value
        gt = self.MO_MapHR.position.value - self.MO_MapHR.rest_position.value

        # Compute metrics only for non-zero displacements
        if np.linalg.norm(gt) > 1e-6:
            error = (gt - pred).reshape(-1)
            self.l2_error.append(np.linalg.norm(error))
            self.MSE_error.append((error.T @ error) / error.shape[0])
            self.l2_deformation.append(np.linalg.norm(gt))
            self.MSE_deformation.append((gt.reshape(-1).T @ gt.reshape(-1)) / gt.shape[0])

    def close(self):

        if len(self.l2_error) > 0:
            print("\nL2 ERROR Statistics :")
            print(f"\t- Distribution : {np.round(np.mean(self.l2_error), 6)} ± {np.round(np.std(self.l2_error), 6)} mm")
            print(f"\t- Extrema : {np.round(np.min(self.l2_error), 6)} -> {np.round(np.max(self.l2_error), 6)} mm")
            relative_error = np.array(self.l2_error) / np.array(self.l2_deformation)
            print(f"\t- Relative Distribution : {np.round(1e2 * relative_error.mean(), 6)} ± {np.round(1e2 * relative_error.std(), 6)} %")
            print(f"\t- Relative Extrema : {np.round(1e2 * relative_error.min(), 6)} -> {np.round(1e2 * relative_error.max(), 6)} %")

            print("\nMSE Statistics :")
            print(f"\t- Distribution : {np.round(np.mean(self.MSE_error), 6)} ± {np.round(np.std(self.MSE_error), 6)} mm²")
            print(f"\t- Extrema : {np.round(np.min(self.MSE_error), 6)} -> {np.round(np.max(self.MSE_error), 6)} mm²")
            relative_error = np.array(self.MSE_error) / np.array(self.MSE_deformation)
            print(f"\t- Relative Distribution : {np.round(1e2 * relative_error.mean(), 6)} ± {np.round(1e2 * relative_error.std(), 6)} %")
            print(f"\t- Relative Extrema : {np.round(1e2 * relative_error.min(), 6)} -> {np.round(1e2 * relative_error.max(), 6)} %")

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
