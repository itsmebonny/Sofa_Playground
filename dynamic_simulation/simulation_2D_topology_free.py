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
sys.path.append(os.path.join(os.path.dirname(__file__), '../simulation'))
from simulation.parameters_2D import p_grid, p_grid_LR
from network.fully_connected_2D import Trainer as Trainer2D
from network.fully_connected import Trainer as Trainer

import SofaCaribou
SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_ROOT'])

class AnimationStepController(Sofa.Core.Controller):
    def __init__(self, node, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.externalForce = [0, -20, 0]
        self.object_mass = 0.05
        self.createGraph(node)
        self.root = node
        self.save = True
        self.l2_error, self.MSE_error = [], []
        self.l2_deformation, self.MSE_deformation = [], []
        self.RMSE_error, self.RMSE_deformation = [], []
        
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
        self.count = 0
        self.inside_counter = 0
        self.directory = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.directory =  self.directory + "_dynamic_simulation"
        self.efficient_sampling = True
        self.magnitudes = np.linspace(10, 40, 20)
        self.angles = np.linspace(0, 2*np.pi, 20)
        self.count_angle = 0
        self.count_magnitude = 0
        self.vector = np.array([np.cos(self.angles[0]), np.sin(self.angles[0]), 0])
        self.versor = self.vector / np.linalg.norm(self.vector)
        self.magnitude = self.magnitudes[0]
        self.externalForce = self.magnitude * self.versor
        self.count_angle += 1
        self.count_magnitude += 1
        if self.save:
            if not os.path.exists('npy'):
                os.mkdir('npy')
            # get current time from computer format yyyy-mm-dd-hh-mm-ss and create a folder with that name
            
            
            os.makedirs(f'npy/{self.directory}')
            print(f"Saving data to npy/{self.directory}")
        self.sampled = False



    def onAnimateBeginEvent(self, event):

        if self.sampled:
            print("================== Sampled all magnitudes and versors ==================\n")
            print ("================== The simulation is over ==================\n")
        if self.count % 5000 == 0:
            self.inside_counter = 0
            self.MO1.position.value = self.MO1.rest_position.value
            self.MO2.position.value = self.MO2.rest_position.value
            if not self.efficient_sampling:
                self.vector = np.random.uniform(-1, 1, 2)
                self.versor = self.vector / np.linalg.norm(self.vector)
                self.magnitude = np.random.uniform(10, 40)
                self.externalForce = np.append(self.magnitude * self.versor, 0)
            else:
                self.magnitude = self.magnitudes[self.count_magnitude % len(self.magnitudes)]
                self.vector = np.array([np.cos(self.angles[self.count_angle % len(self.angles)]), np.sin(self.angles[self.count_angle % len(self.angles)]), 0])
                self.versor = self.vector / np.linalg.norm(self.vector)
                self.externalForce = self.magnitude * self.versor
                self.count_angle += 1
                self.count_magnitude += 1

            
        self.count += 1
        self.inside_counter += 1
        #self.externalForce = [0, -20, 0]

        self.exactSolution.removeObject(self.cff)
        self.cff = self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.LowResSolution.removeObject(self.cffLR)
        self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff.init()
        self.cffLR.init()


        self.start_time = process_time()



    def onAnimateEndEvent(self, event):
        self.end_time = process_time()

        print("Computation time for 1 time step: ", self.end_time - self.start_time)
        U_high = self.compute_displacement(self.MO1_LR)
        U_low = self.compute_displacement(self.MO_training)
        V_high = self.compute_velocity(self.MO1_LR)
        V_low = self.compute_velocity(self.MO_training)
        #save data
        if self.save and ((self.count % 100 == 0 and self.inside_counter > 2500) or (self.count % 25 == 0 and self.inside_counter <= 2500)):
            np.save(f'npy/{self.directory}/HighResPoints_{round(np.linalg.norm(self.externalForce), 3)}_x_{round(self.versor[0], 3)}_y_{round(self.versor[1], 3)}_{int(self.count/100)}.npy', np.array(U_high))
            np.save(f'npy/{self.directory}/CoarseResPoints_{round(np.linalg.norm(self.externalForce), 3)}_x_{round(self.versor[0], 3)}_y_{round(self.versor[1], 3)}_{int(self.count/100)}.npy', np.array(U_low))
            np.save(f'npy/{self.directory}/HighResVel_{round(np.linalg.norm(self.externalForce), 3)}_x_{round(self.versor[0], 3)}_y_{round(self.versor[1], 3)}_{int(self.count/100)}.npy', np.array(V_high))
            np.save(f'npy/{self.directory}/CoarseResVel_{round(np.linalg.norm(self.externalForce), 3)}_x_{round(self.versor[0], 3)}_y_{round(self.versor[1], 3)}_{int(self.count/100)}.npy', np.array(V_low))
            print(f"Saved data for {round(np.linalg.norm(self.externalForce), 3)}_x_{round(self.versor[0], 3)}_y_{round(self.versor[1], 3)}_{int(self.count/100)}")

        
    def compute_displacement(self, mechanical_object):
        # Compute the displacement between the high and low resolution solutions
        U = mechanical_object.position.value.copy() - mechanical_object.rest_position.value.copy()
        return U
    
    def compute_velocity(self, mechanical_object):
        # Compute the velocity of the mechanical object
        V = mechanical_object.velocity.value.copy()
        return V

    def close(self):

        if len(self.l2_error) > 0:
            print("\nL2 ERROR Statistics :")
            print(f"\t- Distribution : {np.round(np.mean(self.l2_error), 4)} ± {np.round(np.std(self.l2_error), 4)} mm")
            print(f"\t- Extrema : {np.round(np.min(self.l2_error), 4)} -> {np.round(np.max(self.l2_error), 4)} mm")
            relative_error = np.array(self.l2_error) / np.array(self.l2_deformation)
            print(f"\t- Relative Distribution : {np.round(1e2 * relative_error.mean(), 4)} ± {np.round(1e2 * relative_error.std(), 4)} %")
            print(f"\t- Relative Extrema : {np.round(1e2 * relative_error.min(), 4)} -> {np.round(1e2 * relative_error.max(), 4)} %")

            print("\nMSE Statistics :")
            print(f"\t- Distribution : {np.round(np.mean(self.MSE_error), 4)} ± {np.round(np.std(self.MSE_error), 4)} mm²")
            print(f"\t- Extrema : {np.round(np.min(self.MSE_error), 4)} -> {np.round(np.max(self.MSE_error), 4)} mm²")
            relative_error = np.array(self.MSE_error) / np.array(self.MSE_deformation)
            print(f"\t- Relative Distribution : {np.round(1e2 * relative_error.mean(), 4)} ± {np.round(1e2 * relative_error.std(), 4)} %")
            print(f"\t- Relative Extrema : {np.round(1e2 * relative_error.min(), 4)} -> {np.round(1e2 * relative_error.max(), 4)} %")
        else:
            print("No data to compute metrics.")


def createScene(rootNode, *args, **kwargs):
    rootNode.dt = 0.005
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
    root=Sofa.Core.Node("root")

if __name__ == '__main__':
    main()
