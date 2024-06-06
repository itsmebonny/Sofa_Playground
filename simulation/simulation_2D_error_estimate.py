import Sofa
import SofaRuntime
import numpy as np 
import os
from Sofa import SofaDeformable
from time import process_time, time
import datetime
from parameters_2D import p_grid, p_grid_LR
# add network path to the python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../network'))

from network.fully_connected import Trainer

import SofaCaribou
SofaRuntime.PluginRepository.addFirstPath(os.environ['CARIBOU_ROOT'])

class AnimationStepController(Sofa.Core.Controller):
    def __init__(self, node, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.externalForce = [0, -10, 0]
        self.createGraph(node)
        self.root = node
        self.save = False

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
        self.exactSolution.addObject('RegularGridTopology', name='grid', min=p_grid.min, max=p_grid.max, nx=p_grid.res[0], ny=p_grid.res[1], nz=p_grid.res[2])
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

        # same object with different resolution

        self.LowResSolution = rootNode.addChild('LowResSolution2D', activated=True)
        self.LowResSolution.addObject('RegularGridTopology', name='grid', min=p_grid_LR.min, max=p_grid_LR.max, nx=p_grid_LR.res[0], ny=p_grid_LR.res[1], nz=p_grid_LR.res[2])
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

        self.LowResSolution.addChild("visual")
        self.LowResSolution.visual.addObject('OglModel', src='@../grid', color='1 0 0 0.2')
        self.LowResSolution.visual.addObject('IdentityMapping', input='@../DOFs', output='@./')

    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        """
        self.inputs = []
        self.outputs = []
        self.save = True
        self.efficient_sampling = True
        if self.efficient_sampling:
            self.count_v = 0
            self.num_versors = 5
            self.versors = self.generate_versors(self.num_versors)
            self.magnitudes = np.linspace(10, 50, 30)
            self.count_m = 0
            self.angles = np.linspace(0, 2*np.pi, self.num_versors, endpoint=False)
            self.starting_points = np.linspace(self.angles[0], self.angles[1], len(self.magnitudes), endpoint=False)
        if self.save:
            if not os.path.exists('npy'):
                os.mkdir('npy')
            # get current time from computer format yyyy-mm-dd-hh-mm-ss and create a folder with that name
            self.directory = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            self.directory = self.directory + "_estimation"
            os.makedirs(f'npy/{self.directory}')
            print(f"Saving data to npy/{self.directory}")
        self.sampled = False



    def onAnimateBeginEvent(self, event):

        # reset positions
        self.MO1.position.value = self.MO1.rest_position.value
        self.MO2.position.value = self.MO2.rest_position.value
        if self.sampled:
            print("================== Sampled all magnitudes and versors ==================\n")
            print ("================== The simulation is over ==================\n")
        
        if not self.efficient_sampling:
            self.vector = np.random.uniform(-1, 1, 2)
            self.versor = self.vector / np.linalg.norm(self.vector)
            self.magnitude = np.random.uniform(10, 50)
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
        self.exactSolution.removeObject(self.cff)
        self.cff = self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff.init()


        self.LowResSolution.removeObject(self.cffLR)
        self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cffLR.init()


        self.start_time = process_time()



    def onAnimateEndEvent(self, event):
        self.end_time = process_time()
        print("Computation time for 1 time step: ", self.end_time - self.start_time)
        print("External force: ", np.linalg.norm(self.externalForce))
        U_high = self.compute_displacement(self.MO1_LR)
        U_low = self.compute_displacement(self.MO2)
        # cut the z component
        # U_high = U_high[:, :2]
        # U_low = U_low[:, :2]
        if self.save and not self.efficient_sampling:    
            np.save(f'npy/{self.directory}/HighResPoints_{round(np.linalg.norm(self.externalForce), 3)}_x_{round(self.versor[0], 3)}_y_{round(self.versor[1], 3)}.npy', np.array(U_high))
            np.save(f'npy/{self.directory}/CoarseResPoints_{round(np.linalg.norm(self.externalForce), 3)}_x_{round(self.versor[0], 3)}_y_{round(self.versor[1], 3)}.npy', np.array(U_low))
        elif self.save and self.efficient_sampling:
            np.save(f'npy/{self.directory}/HighResPoints_{round(self.magnitudes[self.count_m], 3)}_x_{round(self.versors[self.count_v][0], 3)}_y_{round(self.versors[self.count_v][1], 3)}.npy', np.array(U_high))
            np.save(f'npy/{self.directory}/CoarseResPoints_{round(self.magnitudes[self.count_m], 3)}_x_{round(self.versors[self.count_v][0], 3)}_y_{round(self.versors[self.count_v][1], 3)}.npy', np.array(U_low)) 
    def compute_displacement(self, mechanical_object):
        # Compute the displacement between the high and low resolution solutions
        U = mechanical_object.position.value.copy() - mechanical_object.rest_position.value.copy()
        return U
    
    def compute_rest_position(self, mechanical_object):
        # Compute the position of the high resolution solution
        return mechanical_object.rest_position.value.copy()
    
    def generate_versors(self, n=30, starting_point=0):
        """
        Generate evenly distributed versor on the unit circle.
        Change the starting point at every new magnitude
        """
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + starting_point
        versors = np.array([np.cos(angles), np.sin(angles)]).T
        return versors


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
    createScene(root)
    Sofa.Simulation.init(root)

    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(800, 600)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()


if __name__ == '__main__':
    main()
