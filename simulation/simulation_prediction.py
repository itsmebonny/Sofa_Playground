import Sofa
import SofaRuntime
import numpy as np 
import os
from Sofa import SofaDeformable
from time import process_time, time
import datetime
from sklearn.preprocessing import MinMaxScaler
# add network path to the python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../network'))

from network.fully_connected import Trainer
from parameters import p_grid, p_grid_LR

import SofaCaribou
SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_ROOT'])

class AnimationStepController(Sofa.Core.Controller):
    def __init__(self, node, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.externalForce = [0, -10, 0]
        self.createGraph(node)
        self.root = node
        self.save = False
        # print cwd
        self.l2_error, self.MSE_error = [], []
        self.l2_deformation, self.MSE_deformation = [], []
        self.network = Trainer('npy/2024-04-18_15:27:30/train', 32, 0.001, 100)
        self.network.load_model('models/model_2024-04-19_10:11:26.pth')

    def createGraph(self, rootNode):

        rootNode.addObject('RequiredPlugin', name='SofaCaribou')
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Geometry') # Needed to use components [SphereCollisionModel]  
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Projective') # Needed to use components [FixedProjectiveConstraint]  
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.Engine.Select') # Needed to use components [BoxROI]  
        rootNode.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Iterative') # Needed to use components [CGLinearSolver]  
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
        rootNode.addObject('VisualStyle', name="visualStyle", displayFlags="showBehaviorModels showForceFields")
        
        sphereRadius=0.025

        # High resolution grid definition
        self.exactSolution = rootNode.addChild('HighResSolution')
        self.exactSolution.addObject('RegularGridTopology', name='grid', min=p_grid.min, max=p_grid.max, nx=p_grid.res[0], ny=p_grid.res[1], nz=p_grid.res[2])
        self.exactSolution.addObject('HexahedronSetTopologyContainer', name='hexaTopo', src='@grid')
        self.MO1 = self.exactSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        self.exactSolution.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 0')

        # High resolution ODE solver 
        self.exactSolution.addObject('StaticSolver', name='ODE', newton_iterations="20", printLog=True)
        self.exactSolution.addObject('CGLinearSolver', name='linear solver', iterations=500, tolerance=1e-08, threshold=1e-08)

        # High resolution material properties
        self.exactSolution.addObject('NeoHookeanMaterial', young_modulus=5000, poisson_ratio=0.4)
        self.exactSolution.addObject('HyperelasticForcefield', template="Hexahedron", printLog=False)

        # High resolution boundary conditions
        self.exactSolution.addObject('BoxROI', name='ROI', box=p_grid.fixed_box)
        self.exactSolution.addObject('FixedConstraint', indices='@ROI.indices')
        self.exactSolution.addObject('BoxROI', name='ROI2', box=p_grid.size)
        self.cff = self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        
        # visualization
        self.exactSolution.addChild('visual')
        self.exactSolution.visual.addObject('OglModel', src='@../grid', color='0 1 0')
        self.exactSolution.visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')

        # Low resolution grid definition
        self.LRSolution = rootNode.addChild('LowResSolution')
        self.LRSolution.addObject('RegularGridTopology', name='grid', min=p_grid_LR.min, max=p_grid_LR.max, nx=p_grid_LR.res[0], ny=p_grid_LR.res[1], nz=p_grid_LR.res[2])
        self.LRSolution.addObject('HexahedronSetTopologyContainer', name='hexaTopo', src='@grid')
        self.MO2 = self.LRSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        self.LRSolution.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 0')

        # Low resolution ODE solver
        self.LRSolution.addObject('StaticSolver', name='ODE', newton_iterations="20", printLog=True)
        self.LRSolution.addObject('CGLinearSolver', name='linear solver', iterations=500, tolerance=1e-08, threshold=1e-08)

        # Low resolution material properties
        self.LRSolution.addObject('NeoHookeanMaterial', young_modulus=5000, poisson_ratio=0.4)
        self.LRSolution.addObject('HyperelasticForcefield', template="Hexahedron", printLog=False)

        # Low resolution boundary conditions
        self.LRSolution.addObject('BoxROI', name='ROI', box=p_grid_LR.fixed_box)
        self.LRSolution.addObject('FixedConstraint', indices='@ROI.indices')
        self.LRSolution.addObject('BoxROI', name='ROI2', box=p_grid_LR.size)
        self.cffLR = self.LRSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        
        # visualization
        self.LRSolution.addChild('visual')
        self.LRSolution.visual.addObject('OglModel', src='@../grid', color='1 0 0 0.2')
        self.LRSolution.visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')

        # ============== NEURAL NETWORK MODEL ==============
        # In SOFA it's just a MechanicalObject without physical properties, the displacement is computed by the neural network

        self.nnModel = rootNode.addChild('NNModel')
        self.nnModel.addObject('RegularGridTopology', name='grid', min=p_grid.min, max=p_grid.max, nx=p_grid.res[0], ny=p_grid.res[1], nz=p_grid.res[2])
        self.MO_NN = self.nnModel.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        self.nnModel.addObject('HexahedronSetTopologyContainer', name='hexaTopo', src='@grid')
        self.nnModel.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 0')


        # boundary conditions

        self.nnModel.addObject('BoxROI', name='ROI', box=p_grid.fixed_box)
        self.nnModel.addObject('FixedConstraint', indices="@ROI.indices")
        self.nnModel.addObject('BoxROI', name='ROI2', box=p_grid.size)
        self.cffNN = self.nnModel.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")

        # visualization
        self.nnModel.addChild('visual')
        self.nnModel.visual.addObject('OglModel', name='VisualBeam', src='@../DOFs', color='0 0 1 0.5')
        self.nnModel.visual.addObject('BarycentricMapping', input='@../DOFs', output='@VisualBeam')


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
        if self.save:
            if not os.path.exists('npy'):
                os.mkdir('npy')
            # get current time from computer format yyyy-mm-dd-hh-mm-ss and create a folder with that name
            self.directory = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            os.makedirs(f'npy/{self.directory}')
            print(f"Saving data to npy/{self.directory}")



    def onAnimateBeginEvent(self, event):

        # reset positions
        self.MO1.position.value = self.MO1.rest_position.value
        self.MO2.position.value = self.MO2.rest_position.value
        self.vector = np.random.uniform(-1, 1, 3)
        self.versor = self.vector / np.linalg.norm(self.vector)
        self.magnitude = np.random.uniform(10, 30)
        self.externalForce = self.magnitude * self.versor

        self.exactSolution.removeObject(self.cff)
        self.cff = self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.LRSolution.removeObject(self.cffLR)
        self.cffLR = self.LRSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff.init()
        self.cffLR.init()


        self.start_time = process_time()



    def onAnimateEndEvent(self, event):
        self.end_time = process_time()

        inputs = np.reshape(self.MO2.position.value, (1, -1))
        if self.network.normalized:
            scaler = MinMaxScaler()
            inputs = scaler.fit_transform(inputs)

        U = self.network.predict(inputs).cpu().numpy()
        if self.network.normalized:
            U = scaler.inverse_transform(U)
        # reshape U to have the same shape as the position
        U = np.reshape(U, self.high_res_shape)
        
        # compute L2 norm of the prediction error
        error = np.linalg.norm(U - self.MO1.position.value)
        print(f"Prediction error: {error}")
        self.compute_metrics()
        self.MO_NN.position.value =  U
        print("Computation time for 1 time step: ", self.end_time - self.start_time)
        print("External force: ", np.linalg.norm(self.externalForce))

    def compute_metrics(self):
        """
        Compute L2 error and MSE for each sample.
        """

        pred = self.MO_NN.position.value - self.MO_NN.rest_position.value
        gt = self.MO1.position.value - self.MO1.rest_position.value

        # Compute metrics only for non-zero displacements
        error = (gt - pred).reshape(-1)
        self.l2_error.append(np.linalg.norm(error))
        self.MSE_error.append((error.T @ error) / error.shape[0])
        self.l2_deformation.append(np.linalg.norm(gt))
        self.MSE_deformation.append((gt.reshape(-1).T @ gt.reshape(-1)) / gt.shape[0])

    def close(self):

        if len(self.l2_error) > 0:
            print("\nL2 ERROR Statistics :")
            print(f"\t- Distribution : {np.mean(self.l2_error)} ± {np.std(self.l2_error)} mm")
            print(f"\t- Extrema : {np.min(self.l2_error)} -> {np.max(self.l2_error)} mm")
            relative_error = np.array(self.l2_error) / np.array(self.l2_deformation)
            print(f"\t- Relative Distribution : {1e2 * relative_error.mean()} ± {1e2 * relative_error.std()} %")
            print(f"\t- Relative Extrema : {1e2 * relative_error.min()} -> {1e2 * relative_error.max()} %")

            print("\nMSE Statistics :")
            print(f"\t- Distribution : {np.mean(self.MSE_error)} ± {np.std(self.MSE_error)} mm²")
            print(f"\t- Extrema : {np.min(self.MSE_error)} -> {np.max(self.MSE_error)} mm²")
            relative_error = np.array(self.MSE_error) / np.array(self.MSE_deformation)
            print(f"\t- Relative Distribution : {1e2 * relative_error.mean()} ± {1e2 * relative_error.std()} %")
            print(f"\t- Relative Extrema : {1e2 * relative_error.min()} -> {1e2 * relative_error.max()} %")
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
