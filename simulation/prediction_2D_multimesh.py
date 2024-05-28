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

from scipy.interpolate import RBFInterpolator

import SofaCaribou
SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_ROOT'])

class AnimationStepController(Sofa.Core.Controller):
    def __init__(self, node, meshes, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.externalForce = [0, -60, 0]
        self.meshes = meshes
        self.child_nodes = []
        self.mechanical_objects = []
        self.trained_MOs = []
        self.cffLRs = []
        self.nb_nodes = []
        self.createGraph(node)
        self.root = node
        self.save = False
        self.l2_error, self.MSE_error = [], []
        self.l2_deformation, self.MSE_deformation = [], []
        self.network = Trainer('npy_gmsh/2024-05-28_11:10:16_estimation_efficient_183nodes/train', 32, 0.001, 1000)
        # self.network.load_model('models/model_2024-05-22_10:25:12.pth') # efficient
        # self.network.load_model('models/model_2024-05-21_14:58:44.pth') # not efficient
        self.network.load_model('models/model_2024-05-28_11:14:17_noisy_183.pth') # efficient noisy
        self.mesh_errors = []
        
        self.mesh_relative_errors = []
        

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
        self.exactSolution.addObject('TriangleSetTopologyContainer', name='triangleTopo', src='@grid')
        self.MO1 = self.exactSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        # self.exactSolution.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@quadTopo")
        self.exactSolution.addObject('StaticSolver', name='ODE', newton_iterations="20", printLog=True)
        self.exactSolution.addObject('CGLinearSolver', iterations=250, name="linear solver", tolerance="1.0e-6", threshold="1.0e-6") 
        self.exactSolution.addObject('TriangularFEMForceField', name="FEM", youngModulus=5000, poissonRatio=0.4, method="large")
        self.exactSolution.addObject('BoxROI', name='ROI', box=p_grid.fixed_box)
        self.exactSolution.addObject('FixedConstraint', indices='@ROI.indices')
        self.exactSolution.addObject('BoxROI', name='ROI2', box="9.9 -1.1 -0.1 10.1 1.1 0.1")
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

        # loop to add all the meshes
        for nb_nodes, mesh in self.meshes.items():
            self.nb_nodes.append(nb_nodes)
            print(f"Adding mesh with {nb_nodes} nodes")
            self.child_nodes.append(rootNode.addChild(f'LowResSolution2D_{nb_nodes}', activated=True))
            self.child_nodes[-1].addObject('MeshGmshLoader', name='grid', filename=mesh)
            self.child_nodes[-1].addObject('TriangleSetTopologyContainer', name='quadTopo', src='@grid')
            self.mechanical_objects.append(self.child_nodes[-1].addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid'))
            self.child_nodes[-1].addObject('StaticSolver', name='ODE', newton_iterations="20", printLog=True)
            self.child_nodes[-1].addObject('CGLinearSolver', iterations=250, name="linear solver", tolerance="1.0e-6", threshold="1.0e-6")
            self.child_nodes[-1].addObject('TriangularFEMForceField', name="FEM", youngModulus=5000, poissonRatio=0.4, method="large")
            self.child_nodes[-1].addObject('BoxROI', name='ROI', box=p_grid_LR.fixed_box)
            self.child_nodes[-1].addObject('FixedConstraint', indices='@ROI.indices')
            self.child_nodes[-1].addObject('BoxROI', name='ROI2', box="9.9 -1.1 -0.1 10.1 1.1 0.1")
            self.cffLRs.append(self.child_nodes[-1].addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1"))
            self.child_nodes[-1].addChild('CoarseMesh')
            self.child_nodes[-1].CoarseMesh.addObject('RegularGridTopology', name='coarseGrid', min=p_grid_LR.min, max=p_grid_LR.max, nx=p_grid_LR.res[0], ny=p_grid_LR.res[1], nz=p_grid_LR.res[2])
            self.child_nodes[-1].CoarseMesh.addObject('TriangleSetTopologyContainer', name='triangleTopo', src='@coarseGrid')
            self.trained_MOs.append(self.child_nodes[-1].CoarseMesh.addObject('MechanicalObject', name='coarseDOFs', template='Vec3d', src='@coarseGrid'))
            self.child_nodes[-1].CoarseMesh.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 0')
            self.child_nodes[-1].CoarseMesh.addObject('BarycentricMapping', name="mapping", input='@../DOFs', input_topology='@../triangleTopo', output='@coarseDOFs', output_topology='@triangleTopo')

            self.child_nodes[-1].addChild("visual")
            self.child_nodes[-1].visual.addObject('OglModel', src='@../grid', color='1 0 0 0.2')
            self.child_nodes[-1].visual.addObject('IdentityMapping', input='@../DOFs', output='@./')



        self.LowResSolution = rootNode.addChild('LowResSolution2D', activated=True)
        self.LowResSolution.addObject('MeshGmshLoader', name='grid', filename="mesh/rectangle_183.msh")
        self.LowResSolution.addObject('TriangleSetTopologyContainer', name='quadTopo', src='@grid')
        self.MO2 = self.LowResSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        # self.LowResSolution.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@quadTopo")
        self.LowResSolution.addObject('StaticSolver', name='ODE', newton_iterations="20", printLog=True)
        self.LowResSolution.addObject('CGLinearSolver', iterations=250, name="linear solver", tolerance="1.0e-6", threshold="1.0e-6")
        self.LowResSolution.addObject('TriangularFEMForceField', name="FEM", youngModulus=5000, poissonRatio=0.4, method="large")
        self.LowResSolution.addObject('BoxROI', name='ROI', box=p_grid_LR.fixed_box)
        self.LowResSolution.addObject('FixedConstraint', indices='@ROI.indices')
        self.LowResSolution.addObject('BoxROI', name='ROI2', box="9.9 -1.1 -0.1 10.1 1.1 0.1")
        self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")

        self.trained_nodes = self.LowResSolution.addChild('CoarseMesh')
        self.trained_nodes.addObject('RegularGridTopology', name='coarseGrid', min=p_grid_LR.min, max=p_grid_LR.max, nx=p_grid_LR.res[0], ny=p_grid_LR.res[1], nz=p_grid_LR.res[2])
        self.trained_nodes.addObject('TriangleSetTopologyContainer', name='triangleTopo', src='@coarseGrid')
        self.MO_training = self.trained_nodes.addObject('MechanicalObject', name='coarseDOFs', template='Vec3d', src='@coarseGrid')
        self.trained_nodes.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 0')
        self.trained_nodes.addObject('BarycentricMapping', name="mapping", input='@../DOFs', input_topology='@../triangleTopo', output='@coarseDOFs', output_topology='@triangleTopo')

        self.LowResSolution.addChild("visual")
        self.LowResSolution.visual.addObject('OglModel', src='@../grid', color='1 0 0 0.2')
        self.LowResSolution.visual.addObject('IdentityMapping', input='@../DOFs', output='@./')








        
        # ============== NEURAL NETWORK MODEL ==============
        # In SOFA it's just a MechanicalObject without physical properties, the displacement is computed by the neural network

        # self.nnModel = rootNode.addChild('NNModel')
        # self.nnModel.addObject('MeshGmshLoader', name='grid', filename='mesh/rectangle_80.msh')
        # self.MO_NN = self.nnModel.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        # self.nnModel.addObject('TriangleSetTopologyContainer', name='quadTopo', src='@grid')
        # # self.nnModel.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 0')


        # # boundary conditions

        # self.nnModel.addObject('BoxROI', name='ROI', box=p_grid.fixed_box)
        # self.nnModel.addObject('FixedConstraint', indices="@ROI.indices")
        # self.nnModel.addObject('BoxROI', name='ROI2', box=p_grid.size)
        # self.cffNN = self.nnModel.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")

        # # visualization
        # self.nnModel.addChild('visual')
        # self.nnModel.visual.addObject('OglModel', name='VisualBeam', src='@../DOFs', color='0 0 1 0.2')
        # self.nnModel.visual.addObject('IdentityMapping', input='@../DOFs', output='@VisualBeam')


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
        if self.save:
            if not os.path.exists('npy'):
                os.mkdir('npy')
            # get current time from computer format yyyy-mm-dd-hh-mm-ss and create a folder with that name
            self.directory = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            os.makedirs(f'npy/{self.directory}')
            print(f"Saving data to npy/{self.directory}")
        
        print("=================== Simulation initialized. ======================")



    def onAnimateBeginEvent(self, event):

        print("======================= Simulation started. =========================")
        # reset positions
        self.MO1.position.value = self.MO1.rest_position.value
        self.MO2.position.value = self.MO2.rest_position.value
        for mo in self.mechanical_objects:
            mo.position.value = mo.rest_position.value
            
        # self.MO_NN.position.value = self.MO_NN.rest_position.value
        
        self.vector = np.random.uniform(-1, 1, 2)
        self.versor = self.vector / np.linalg.norm(self.vector)
        self.magnitude = np.random.uniform(10, 80)
        self.externalForce = np.append(self.magnitude * self.versor, 0)

        # self.externalForce = [0, -40, 0]
        # self.externalForce_LR = [0, -60, 0]

        self.exactSolution.removeObject(self.cff)
        self.cff = self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff.init()
        i = 0
        for cffLR in self.cffLRs:
            self.child_nodes[i].removeObject(cffLR)
            cffLR = self.child_nodes[i].addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
            cffLR.init()
            self.cffLRs[i] = cffLR
            i += 1

        self.LowResSolution.removeObject(self.cffLR)
        self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cffLR.init()


        self.start_time = process_time()



    def onAnimateEndEvent(self, event):
        self.end_time = process_time()

        coarse_pos = self.MO_training.position.value.copy() - self.MO_training.rest_position.value.copy()

        inputs = np.reshape(coarse_pos, -1)
        
        U = self.network.predict(inputs).cpu().numpy()
        U = np.reshape(U, (self.low_res_shape[0], self.low_res_shape[1]))
 
        self.MO_training.position.value = self.MO_training.position.value + U
        self.mesh_errors.append([])
        self.mesh_relative_errors.append([])
        
        # ========= UPDATE ALL THE MESHES =========
        for nb_nodes, mo in zip(self.meshes.keys(), self.trained_MOs):
            
            coarse_pos = mo.position.value.copy() - mo.rest_position.value.copy()
            inputs = np.reshape(coarse_pos, -1)
            U = self.network.predict(inputs).cpu().numpy()
            U = np.reshape(U, (self.low_res_shape[0], self.low_res_shape[1]))
            mo.position.value = mo.position.value + U

            l2_error, mse_error, l2_def, mse_def = self.compute_errors(mo)
    
            self.mesh_errors[-1].append(l2_error)
            self.mesh_relative_errors[-1].append(l2_error / l2_def)



            
    # positions = self.MO_training.position.value.copy()
    # print("Positions: ", positions.shape)
    # displacement = self.MO_training.position.value.copy() - self.MO_training.rest_position.value.copy()
    # interpolator = RBFInterpolator(positions, displacement, neighbors=5)
    # interpolate_positions = self.MO2.position.value.copy()
    # corrected_displacement = interpolator(interpolate_positions)
    # print("Corrected displacement: ", corrected_displacement)

    # self.MO2.position.value = self.MO2.rest_position.value + corrected_displacement


        # ============== UPDATE THE NN MODEL ==============
        # self.MO_NN.position.value = self.MO2.position.value + U


        # error = np.linalg.norm(self.MO_NN.position.value - self.MO1.position.value)
        # print(f"Prediction error: {error}")
        self.compute_metrics()
        print("Computation time for 1 time step: ", self.end_time - self.start_time)
        print("External force: ", np.linalg.norm(self.externalForce))

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

    def compute_errors(self, mechanical_object):
        """
        Compute L2 error and MSE for each sample.
        """

        pred = mechanical_object.position.value - mechanical_object.rest_position.value
        gt = self.MO1_LR.position.value - self.MO1_LR.rest_position.value

        l2_error = np.linalg.norm(gt - pred)
        mse_error = (gt - pred).T @ (gt - pred) / gt.shape[0]
        l2_def = np.linalg.norm(gt)
        mse_def = (gt.reshape(-1).T @ gt.reshape(-1)) / gt.shape[0]

        return l2_error, mse_error, l2_def, mse_def
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

            return self.nb_nodes, self.mesh_errors, self.mesh_relative_errors

        else:
            print("No data to compute metrics.")

    
        

        

def createScene(rootNode, mesh_name, *args, **kwargs):
    rootNode.dt = 0.01
    rootNode.gravity = [0, 0, 0]
    rootNode.name = 'root'
    asc = AnimationStepController(rootNode, mesh_name, *args, **kwargs)
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

    # create a list of mesh files with rectangle_xx.msh format, chech that the filename has rectangle in it
    meshes = [f for f in os.listdir('mesh') if 'rectangle' in f and f.endswith('.msh')]
    # add prefix to the mesh files to have the full path
    meshes = [os.path.join('mesh', f) for f in meshes]
    # create a dict with the number of nodes as key and the mesh file as value
    meshes = {int(f.split('_')[1].split('.')[0]): f for f in meshes}
    # sort the dict by the number of nodes
    meshes = dict(sorted(meshes.items()))
    print("Meshes: ", meshes)
    # create a new dictioanry with the keys below 1000
    meshes = {k: v for k, v in meshes.items() if k < 1000}
    
   
    root=Sofa.Core.Node("root")
    rootNode, asc = createScene(root, meshes)  # rewrite the graph scene adding a node for each mesh
    Sofa.Simulation.init(root)

    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(800, 600)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()
    nb_nodes, errors, relative_error = asc.close()
    
    print("Errors: ", errors)
    print("Relative errors: ", relative_error)
    print(f"Length of errors: {len(errors)}")
    print(f"Length of relative errors: {len(relative_error)}")
    print(f"First length of errors: {len(errors[0])}")
    #plot the errors
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(nb_nodes, np.mean(errors, axis=0), label='L2 error')
    ax[0].set_xlabel('Number of nodes')
    ax[0].set_ylabel('L2 error')
    ax[0].set_title('L2 error vs number of nodes')
    ax[0].legend()
    ax[1].plot(nb_nodes, np.mean(relative_error, axis=0)*100, label='Relative error')
    ax[1].set_xlabel('Number of nodes')
    ax[1].set_ylabel('Relative error')
    ax[1].set_title('Relative error vs number of nodes')
    ax[1].legend()
    plt.show()

if __name__ == '__main__':
    main()
