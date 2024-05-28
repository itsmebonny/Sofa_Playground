import Sofa
import SofaRuntime
import numpy as np 
from time import process_time
from Sofa import SofaDeformable 

class AnimationStepController(Sofa.Core.Controller):
    def __init__(self, node, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.externalForce = [0, -10, 0]
        self.createGraph(node)
        self.root = node

    def createGraph(self, rootNode):

        plugins = ['MultiThreading', 'SofaCUDA', 'Sofa.Component.Constraint.Projective', 'Sofa.Component.Engine.Select', 'Sofa.Component.LinearSolver.Iterative', 'Sofa.Component.Mapping.Linear', 'Sofa.Component.MechanicalLoad', 'Sofa.Component.ODESolver.Backward', 'Sofa.Component.SolidMechanics.FEM.Elastic', 'Sofa.Component.StateContainer', 'Sofa.Component.Topology.Container.Dynamic', 'Sofa.Component.Topology.Container.Grid', 'Sofa.Component.Visual']
        for plugin in plugins:
            rootNode.addObject('RequiredPlugin', name=plugin)

        rootNode.addObject('DefaultAnimationLoop')
        rootNode.addObject('DefaultVisualManagerLoop')
        rootNode.addObject('VisualStyle', name="visualStyle", displayFlags="showBehaviorModels showForceFields showCollisionModels")
        
        sphereRadius=0.025

        self.exactSolution = rootNode.addChild('HighResSolution')
        self.exactSolution.addObject('RegularGridTopology', n="50 10 10", min="0 -1 -1", max='10 1 1', name='grid')
        self.exactSolution.addObject('HexahedronSetTopologyContainer', name='hexaTopo', src='@grid')
        self.MO1 = self.exactSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        self.exactSolution.addObject('StaticSolver', name='ODE', newton_iterations="20", printLog=True)
        self.exactSolution.addObject('ParallelCGLinearSolver', template="ParallelCompressedRowSparseMatrixMat3x3d", iterations=500, tolerance=1e-08, threshold=1e-08, warmStart=True) 
        self.exactSolution.addObject('ParallelHexahedronFEMForceField', name="FEM", youngModulus="5000", poissonRatio="0.45", method="large", updateStiffnessMatrix="false")
        self.exactSolution.addObject('BoxROI', name='ROI', box='-0.1 -2.1 -2.1 0.1 2.1 2.1')
        self.exactSolution.addObject('FixedConstraint', indices='@ROI.indices')
        self.exactSolution.addObject('BoxROI', name='ROI2', box='9.9 -1.1 -1.1 10.1 2.1 1.1')
        self.cff = self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.exactSolution.addChild('visual')
        self.exactSolution.visual.addObject('OglModel', src='@../grid', color='0 0 1')
        self.exactSolution.visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')

        self.LRSolution = rootNode.addChild('LowResSolution')
        self.LRSolution.addObject('RegularGridTopology', n="25 5 5", min="0 -1 -1", max='10 1 1', name='grid')
        self.LRSolution.addObject('HexahedronSetTopologyContainer', name='hexaTopo', src='@grid')
        self.MO2 = self.LRSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        self.LRSolution.addObject('StaticSolver', name='ODE', newton_iterations="20", printLog=True)
        self.LRSolution.addObject('ParallelCGLinearSolver', template="ParallelCompressedRowSparseMatrixMat3x3d", iterations=500, tolerance=1e-08, threshold=1e-08, warmStart=True) 
        self.LRSolution.addObject('ParallelHexahedronFEMForceField', name="FEM", youngModulus="5000", poissonRatio="0.45", method="large", updateStiffnessMatrix="false")
        self.LRSolution.addObject('BoxROI', name='ROI', box='-0.1 -2.1 -2.1 0.1 2.1 2.1')
        self.LRSolution.addObject('FixedConstraint', indices='@ROI.indices')
        self.LRSolution.addObject('BoxROI', name='ROI2', box='9.9 -1.1 -1.1 10.1 2.1 1.1')
        self.cffLR = self.LRSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.LRSolution.addChild('visual')
        self.LRSolution.visual.addObject('OglModel', src='@../grid', color='1 0 0 0.2')
        self.LRSolution.visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')


    def onAnimateBeginEvent(self, event):

        self.MO1.position.value = self.MO1.rest_position.value
        self.MO2.position.value = self.MO2.rest_position.value


        self.vector = np.random.uniform(-1, 1, 3)
        self.versor = self.vector / np.linalg.norm(self.vector)
        self.magnitude = np.random.uniform(20, 50)
        self.externalForce = self.magnitude * self.versor

        self.exactSolution.removeObject(self.cff)
        self.cff = self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff.init()

        self.LRSolution.removeObject(self.cffLR)
        self.cffLR = self.LRSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cffLR.init()

        self.start_time = process_time()

    def onAnimateEndEvent(self, event):
        self.end_time = process_time()
        print("Computation time for 1 time step: ", self.end_time - self.start_time)

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
