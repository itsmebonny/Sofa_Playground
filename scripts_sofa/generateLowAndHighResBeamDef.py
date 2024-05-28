import Sofa
import SofaRuntime
import numpy as np 
import os
from Sofa import SofaDeformable
from time import process_time

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

        plugins = ['SofaCaribou', 'Sofa.Component.Collision.Geometry', 'Sofa.Component.Constraint.Projective', 'Sofa.Component.Engine.Select', 'Sofa.Component.LinearSolver.Iterative', 'Sofa.Component.Mapping.Linear', 'Sofa.Component.MechanicalLoad', 'Sofa.Component.ODESolver.Backward', 'Sofa.Component.SolidMechanics.FEM.Elastic', 'Sofa.Component.StateContainer', 'Sofa.Component.Topology.Container.Dynamic', 'Sofa.Component.Topology.Container.Grid', 'Sofa.Component.Visual']
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
        self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        
        # visualization
        self.exactSolution.addChild('visual')
        self.exactSolution.visual.addObject('OglModel', src='@../grid', color='0 0 1')
        self.exactSolution.visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')

        self.exactSolution = rootNode.addChild('LowResSolution')
        self.exactSolution.addObject('RegularGridTopology', n="25 5 5", min="0 -1 -1", max='10 1 1', name='grid')
        self.exactSolution.addObject('HexahedronSetTopologyContainer', name='hexaTopo', src='@grid')
        self.MO2 = self.exactSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        self.exactSolution.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 0')
        self.exactSolution.addObject('StaticSolver', name='ODE', newton_iterations="20", printLog=True)
        self.exactSolution.addObject('CGLinearSolver', name='linear solver', iterations=500, tolerance=1e-08, threshold=1e-08)
        self.exactSolution.addObject('NeoHookeanMaterial', young_modulus=5000, poisson_ratio=0.4)
        self.exactSolution.addObject('HyperelasticForcefield', template="Hexahedron", printLog=False)
        self.exactSolution.addObject('BoxROI', name='ROI', box='-0.1 -2.1 -2.1 0.1 2.1 2.1')
        self.exactSolution.addObject('FixedConstraint', indices='@ROI.indices')
        self.exactSolution.addObject('BoxROI', name='ROI2', box='9.9 -1.1 -1.1 10.1 2.1 1.1')
        self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.exactSolution.addChild('visual')
        self.exactSolution.visual.addObject('OglModel', src='@../grid', color='1 0 0 0.2')
        self.exactSolution.visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')

    def onAnimateBeginEvent(self, event):
        self.start_time = process_time()



    def onAnimateEndEvent(self, event):
        self.end_time = process_time()
        print("Computation time for 1 time step: ", self.end_time - self.start_time)
        if self.save:    
            if not os.path.exists('npy'):
                os.mkdir('npy')
            np.save('npy/HighResPoints.npy', self.MO1.position.value)
            np.save('npy/CoarseResPoints.npy', self.MO2.position.value)

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
