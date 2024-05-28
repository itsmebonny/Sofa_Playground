import Sofa
import SofaRuntime
import numpy as np 
import os
from Sofa import SofaDeformable
from time import process_time, time
import datetime
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
        rootNode.addObject('VisualStyle', name="visualStyle", displayFlags="showBehaviorModels showForceFields showCollisionModels")
        
        sphereRadius=0.025

        # High resolution grid definition
        self.exactSolution = rootNode.addChild('HighResSolution')
        self.exactSolution.addObject('RegularGridTopology', n="50 10 10", min="0 -1 -1", max='10 1 1', name='grid')
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
        self.exactSolution.addObject('BoxROI', name='ROI', box='-0.1 -2.1 -2.1 0.1 2.1 2.1')
        self.exactSolution.addObject('FixedConstraint', indices='@ROI.indices')
        self.exactSolution.addObject('BoxROI', name='ROI2', box='9.9 -1.1 -1.1 10.1 2.1 1.1')
        self.cff = self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        
        # visualization
        self.exactSolution.addChild('visual')
        self.exactSolution.visual.addObject('OglModel', src='@../grid', color='0 0 1')
        self.exactSolution.visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')

        # Low resolution grid definition
        self.LRSolution = rootNode.addChild('LowResSolution')
        self.LRSolution.addObject('RegularGridTopology', n="25 5 5", min="0 -1 -1", max='10 1 1', name='grid')
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
        self.LRSolution.addObject('BoxROI', name='ROI', box='-0.1 -2.1 -2.1 0.1 2.1 2.1')
        self.LRSolution.addObject('FixedConstraint', indices='@ROI.indices')
        self.LRSolution.addObject('BoxROI', name='ROI2', box='9.9 -1.1 -1.1 10.1 2.1 1.1')
        self.cffLR = self.LRSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        
        # visualization
        self.LRSolution.addChild('visual')
        self.LRSolution.visual.addObject('OglModel', src='@../grid', color='1 0 0 0.2')
        self.LRSolution.visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')

        self.force_steps = 20
        self.directions = 30

        self.versors = self.generate_versors(self.directions)
        self.minForce = 10
        self.maxForce = 30
        self.magnitudes = np.linspace(self.minForce, self.maxForce, self.force_steps)

    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        """
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
        self.count_v = 0
        self.count_m = 0
        self.sphere_sampled = False


    def onAnimateBeginEvent(self, event):
        if self.sphere_sampled:
            self.close()

        # reset positions
        self.MO1.position.value = self.MO1.rest_position.value
        self.MO2.position.value = self.MO2.rest_position.value
        self.vector = self.versors[self.count_v]
        self.magnitude = self.magnitudes[self.count_m]
        self.externalForce = self.magnitude * self.vector


        self.exactSolution.removeObject(self.cff)
        self.cff = self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.LRSolution.removeObject(self.cffLR)
        self.cffLR = self.LRSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff.init()
        self.cffLR.init()

        self.count_m += 1
        if self.count_m == self.force_steps:
            self.count_m = 0
            self.count_v += 1
            if self.count_v == self.directions:
                self.count_v = 0
                self.sphere_sampled = True
        

        self.start_time = process_time()



    def onAnimateEndEvent(self, event):
        self.end_time = process_time()
        print("Computation time for 1 time step: ", self.end_time - self.start_time)
        print(f"Sample number: {self.count_m + (self.count_v * self.force_steps)}")
        sample = self.count_m + (self.count_v * self.force_steps)
        U_high = self.compute_displacement(self.MO1)
        U_low = self.compute_displacement(self.MO2)
        if self.save:    
            np.save(f'npy/{self.directory}/HighResPoints_{sample}.npy', np.array(U_high))
            np.save(f'npy/{self.directory}/CoarseResPoints_{sample}.npy', np.array(U_low))
            print(f"Saved data for external force {round(np.linalg.norm(self.externalForce), 3)}")
    
    def compute_displacement(self, mechanical_object):
        # Compute the displacement between the high and low resolution solutions
        U = mechanical_object.position.value.copy() - mechanical_object.rest_position.value.copy()
        return U
    
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
