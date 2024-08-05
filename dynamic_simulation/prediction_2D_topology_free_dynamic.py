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
from network.FC_Error_estimation_dynamic import Trainer as Trainer
from parameters_2D import p_grid, p_grid_LR, p_grid_test

from scipy.interpolate import RBFInterpolator, griddata

import SofaCaribou
SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_ROOT'])

class AnimationStepController(Sofa.Core.Controller):
    def __init__(self, node, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.externalForce = [0, -30, 0]
        self.object_mass = 1
        self.createGraph(node)
        self.root = node
        self.save = False
        self.l2_error, self.MSE_error = [], []
        self.l2_deformation, self.MSE_deformation = [], []
        self.RMSE_error, self.RMSE_deformation = [], []
        self.RRMSE_error, self.RRMSE_deformation = [], []
        self.network = Trainer('npy/2024-07-30_09:12:39_dynamic_simulation/train', 32, 0.001, 500)
        # self.network.load_model('models/model_2024-05-22_10:25:12.pth') # efficient
        # self.network.load_model('models/model_2024-05-21_14:58:44.pth') # not efficient
        self.network.load_model('models/model_2024-08-01_10:18:47_dynamic.pth') # efficient noisy

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
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.IO.Mesh') # Needed to use components [MeshGmshLoader]

        rootNode.addObject('DefaultAnimationLoop')
        rootNode.addObject('DefaultVisualManagerLoop') 
        rootNode.addObject('VisualStyle', name="visualStyle", displayFlags="showBehaviorModels showCollisionModels")
        
        sphereRadius=0.025


        self.exactSolution = rootNode.addChild('HighResSolution2D', activated=True)
        self.exactSolution.addObject('MeshGmshLoader', name='grid', filename='mesh/rectangle_1166.msh')
        self.surface_topo = self.exactSolution.addObject('TriangleSetTopologyContainer', name='triangleTopo', src='@grid')
        self.MO1 = self.exactSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        self.exactSolution.addObject('MeshMatrixMass', totalMass=self.object_mass, name="SparseMass", topology="@triangleTopo")
        self.exactSolution.addObject('EulerImplicitSolver', name="ODEsolver", rayleighStiffness=0, rayleighMass=0)
        self.exactSolution.addObject('CGLinearSolver', iterations=1000, name="linear solver", tolerance="1.0e-8", threshold="1.0e-8") 
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
        self.LowResSolution.addObject('MeshMatrixMass', totalMass=self.object_mass, name="SparseMass", topology="@quadTopo")
        self.LowResSolution.addObject('EulerImplicitSolver', name="ODEsolver", rayleighStiffness=0, rayleighMass=0)
        self.LowResSolution.addObject('CGLinearSolver', iterations=1000, name="linear solver", tolerance="1.0e-8", threshold="1.0e-8") 
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

        self.LowResSolution.addChild("visual_noncorrected")
        self.LowResSolution.visual_noncorrected.addObject('OglModel', src='@../grid', color='0 1 0 0.5')
        self.LowResSolution.visual_noncorrected.addObject('IdentityMapping', input='@../DOFs', output='@./')

        self.nb_nodes = len(self.MO1.position.value)
        self.nb_nodes_LR = len(self.MO1_LR.position.value)

        self.high_res_shape = np.array((p_grid.nb_nodes, 3))
        self.low_res_shape = np.array((p_grid_LR.nb_nodes, 3))


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
        self.timestep = 0
        if self.save:
            if not os.path.exists('npy'):
                os.mkdir('npy')
            # get current time from computer format yyyy-mm-dd-hh-mm-ss and create a folder with that name
            self.directory = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            os.makedirs(f'npy/{self.directory}')
            print(f"Saving data to npy/{self.directory}")

        surface = self.surface_topo
        surface_LR = self.surface_topo_LR

        self.idx_surface = surface.triangles.value.reshape(-1)
        self.idx_surface_LR = surface_LR.triangles.value.reshape(-1)
        self.angles = np.linspace(0, 2*np.pi, 20)
        



    def onAnimateBeginEvent(self, event):

        # reset positions
        # 
        # self.MO_NN.position.value = self.MO_NN.rest_position.value

        if self.timestep % 5000 == 0:
            self.MO1.position.value = self.MO1.rest_position.value
            self.MO2.position.value = self.MO2.rest_position.value

            indx = np.random.randint(0, 20)

            self.theta = np.random.uniform(0, 2*np.pi)
            self.vector = np.array([np.cos(self.angles[indx]), np.sin(self.angles[indx]), 0])
            self.versor = self.vector / np.linalg.norm(self.vector)
            #self.versor = np.array([np.cos(self.theta), np.sin(self.theta)])
            self.magnitude = np.random.uniform(10, 40)
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
                
            # # Set the new bounding box
            # self.exactSolution.removeObject(self.cff_box)
            # self.cff_box = self.exactSolution.addObject('BoxROI', name='ForceBox', drawBoxes=False, drawSize=1,
            #                                     box=[x_min, y_min, -0.1, x_max, y_max, 0.1])
            # self.cff_box.init()

            # self.LowResSolution.removeObject(self.cff_box_LR)
            # self.cff_box_LR = self.LowResSolution.addObject('BoxROI', name='ForceBox', drawBoxes=True, drawSize=1,
            #                                     box=[x_min, y_min, -0.1, x_max, y_max, 0.1])
            # self.cff_box_LR.init()

            # # Get the intersection with the surface
            # indices = list(self.cff_box.indices.value)
            # indices = list(set(indices).intersection(set(self.idx_surface)))
            # indices_LR = list(self.cff_box_LR.indices.value)
            # indices_LR = list(set(indices_LR).intersection(set(self.idx_surface_LR)))
            self.exactSolution.removeObject(self.cff)
            self.cff = self.exactSolution.addObject('ConstantForceField', indices='@ROI2.indices', totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
            self.cff.init()


            self.LowResSolution.removeObject(self.cffLR)
            self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices='@ROI2.indices', totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
            self.cffLR.init()
        
        # if (self.timestep - 50) % 100 == 0:
        #     # indices = list(self.cff_box.indices.value)
        #     # indices = list(set(indices).intersection(set(self.idx_surface)))
        #     # indices_LR = list(self.cff_box_LR.indices.value)
        #     # indices_LR = list(set(indices_LR).intersection(set(self.idx_surface_LR)))
        #     self.exactSolution.removeObject(self.cff)
        #     self.cff = self.exactSolution.addObject('ConstantForceField', indices='@ROI2.indices', totalForce=[0, 0, 0], showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        #     self.cff.init()


        #     self.LowResSolution.removeObject(self.cffLR)
        #     self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices='@ROI2.indices', totalForce=[0, 0, 0], showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        #     self.cffLR.init()

        # self.MO_training.reset_velocity = np.zeros_like(self.MO_training.velocity.value)
        # self.MO2.reset_velocity = np.zeros_like(self.MO2.velocity.value)


        self.start_time = process_time()



    def onAnimateEndEvent(self, event):
        

        coarse_pos = self.MO_training.position.value.copy() - self.MO_training.rest_position.value.copy()
        coarse_vel = self.MO_training.velocity.value.copy()

        coarse_pos_res = np.reshape(coarse_pos, -1)
        coarse_vel_res = np.reshape(coarse_vel, -1)

        
        #print("Coarse position: ", coarse_pos.shape)
        # cut the z component
        # coarse_pos = coarse_pos[:, :2]
        # print("Coarse position shape: ", coarse_pos.shape)
        inputs = np.concatenate((coarse_pos_res, coarse_vel_res))
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

        # PREDICTION
        self.MO_training.position.value = self.MO_training.position.value + U
        
        

        # U_high = self.compute_displacement(self.MO1)
        # U_low = self.compute_displacement(self.MO2)
        # U_training = self.compute_displacement(self.MO_training)

        # rest_position_high = self.compute_rest_position(self.MO1)
        # rest_position_low = self.compute_rest_position(self.MO2)
        # rest_postion_training = self.compute_rest_position(self.MO_training)

        # if not os.path.exists('interpolation_tests'):
        #     os.mkdir('interpolation_tests')
        # np.save(f'interpolation_tests/U_high.npy', U_high)
        # np.save(f'interpolation_tests/U_low.npy', U_low)
        # np.save(f'interpolation_tests/U_training.npy', U_training)
        # np.save(f'interpolation_tests/rest_position_high.npy', rest_position_high)
        # np.save(f'interpolation_tests/rest_position_low.npy', rest_position_low)
        # np.save(f'interpolation_tests/rest_position_training.npy', rest_postion_training)

            # err = np.linalg.norm(self.MO1_LR.position.value - self.MO_training.position.value)
            # print(f"Prediction error: {err}")
            # self.errs.append(err)
                
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


        #PREDICTION 
        self.visual_model.position.value = interpolate_positions + corrected_displacement



        self.end_time = process_time()
        # error = np.linalg.norm(self.MO_NN.position.value - self.MO1.position.value)
        # print(f"Prediction error: {error}")
        self.compute_metrics()
        print("Computation time for 1 time step: ", self.end_time - self.start_time)
        print("External force: ", np.linalg.norm(self.externalForce))
        print("L2 error: ", self.l2_error[-1])
        print("L2 deformation: ", self.l2_deformation[-1])
        print("Relative error: ", self.l2_error[-1]/np.linalg.norm(self.MO_training.position.value - self.MO_training.rest_position.value))

        self.timestep += 1

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

        #ADD Relative RMSE
        self.RRMSE_error.append(np.sqrt(((error.T @ error) / error.shape[0]) / ((gt.reshape(-1).T @ gt.reshape(-1)))))


    def compute_displacement(self, mechanical_object):
        # Compute the displacement between the high and low resolution solutions
        U = mechanical_object.position.value.copy() - mechanical_object.rest_position.value.copy()
        return U
    
    def compute_rest_position(self, mechanical_object):
        # Compute the position of the high resolution solution
        return mechanical_object.rest_position.value.copy()


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
