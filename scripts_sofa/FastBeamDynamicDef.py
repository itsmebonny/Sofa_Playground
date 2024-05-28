import Sofa
import SofaRuntime
import numpy as np 
from time import process_time
from Sofa import SofaDeformable 

class AnimationStepController(Sofa.Core.Controller):
    def __init__(self, node, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.externalForce = [0, -0.5, 0]
        self.createGraph(node)
        self.root = node

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
        self.exactSolution.addObject('RegularGridTopology', n="50 10 1", min="0 -1 0", max='10 1 0', name='grid')
        self.exactSolution.addObject('TriangleSetTopologyContainer', name='quadTopo', src='@grid')
        self.exactSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        self.exactSolution.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@quadTopo")
        self.exactSolution.addObject('EulerImplicitSolver', name="ODEsolver", rayleighStiffness=0, rayleighMass=0)
        self.exactSolution.addObject('CGLinearSolver', iterations=250, name="linear solver", tolerance="1.0e-6", threshold="1.0e-6") 
        self.exactSolution.addObject('TriangularFEMForceField', name="FEM", youngModulus=10000, poissonRatio=0.4, method="large")
        self.exactSolution.addObject('BoxROI', name='ROI', box='-0.1 -2.1 -2.1 0.1 2.1 2.1')
        self.exactSolution.addObject('FixedConstraint', indices='@ROI.indices')

        self.exactSolution.addChild('visual')
        self.exactSolution.visual.addObject('OglModel', src='@../grid', color='1 0 0')
        self.exactSolution.visual.addObject('IdentityMapping', name="identityMapping", input="@../DOFs", output="@./")

        self.LowResSolution = rootNode.addChild('LowResSolution2D', activated=True)
        self.LowResSolution.addObject('RegularGridTopology', n="20 5 1", min="0 -1 0", max='10 1 0', name='grid')
        self.LowResSolution.addObject('TriangleSetTopologyContainer', name='quadTopo', src='@grid')
        self.LowResSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        self.LowResSolution.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@quadTopo")
        self.LowResSolution.addObject('EulerImplicitSolver', name="ODEsolver", rayleighStiffness=0, rayleighMass=0)
        self.LowResSolution.addObject('CGLinearSolver', iterations=250, name="linear solver", tolerance="1.0e-6", threshold="1.0e-6")
        self.LowResSolution.addObject('TriangularFEMForceField', name="FEM", youngModulus=8510, poissonRatio=0.4, method="large")
        self.LowResSolution.addObject('BoxROI', name='ROI2', box='-0.1 -2.1 -2.1 0.1 2.1 2.1')
        self.LowResSolution.addObject('FixedConstraint', indices='@ROI2.indices')

        self.LowResSolution.addChild('visual')
        self.LowResSolution.visual.addObject('OglModel', src='@../grid', color='0 0 1 0.5')
        self.LowResSolution.visual.addObject('IdentityMapping', name="identityMapping", input="@../DOFs", output="@./")

        # self.exactSolution = rootNode.addChild('HighResSolution', activated=False)
        # self.exactSolution.addObject('VisualStyle', name="visualStyle", displayFlags="showWireframe")
        # self.exactSolution.addObject('RegularGridTopology', n="50 10 10", min="0 -1 -1", max='10 1 1', name='grid')
        # self.exactSolution.addObject('HexahedronSetTopologyContainer', name='hexaTopo', src='@grid')
        # self.exactSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        # self.exactSolution.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@hexaTopo")
        # self.exactSolution.addObject('EulerImplicitSolver', name="ODEsolver", rayleighStiffness=0, rayleighMass=0)
        # self.exactSolution.addObject('SparseLDLSolver', template="CompressedRowSparseMatrixMat3x3d") 
        # self.exactSolution.addObject('NeoHookeanMaterial', young_modulus=5000, poisson_ratio=0.4)
        # self.exactSolution.addObject('HyperelasticForcefield', template="Hexahedron", printLog=True)
        # self.exactSolution.addObject('BoxROI', name='ROI', box='-0.1 -2.1 -2.1 0.1 2.1 2.1')
        # self.exactSolution.addObject('FixedConstraint', indices='@ROI.indices')
        # #self.exactSolution.addObject('BoxROI', name='ROI2', box='9.9 -1.1 -1.1 10.1 2.1 1.1')
        # #self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", forces=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")

        # self.coarseSolution = rootNode.addChild('LowResSolution', activated=False)
        # self.coarseSolution.addObject('RegularGridTopology', n="25 5 5", min="0 -1 -1", max='10 1 1', name='grid')
        # self.coarseSolution.addObject('HexahedronSetTopologyContainer', name='hexaTopo', src='@grid')
        # self.coarseSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        # self.coarseSolution.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@hexaTopo")
        # self.coarseSolution.addObject('EulerImplicitSolver', name="ODEsolver", rayleighStiffness=0, rayleighMass=0)
        # self.coarseSolution.addObject('SparseLDLSolver', template="CompressedRowSparseMatrixMat3x3d")
        # self.coarseSolution.addObject('NeoHookeanMaterial', young_modulus=5000, poisson_ratio=0.4)
        # self.coarseSolution.addObject('HyperelasticForcefield', template="Hexahedron", printLog=True)
        # self.coarseSolution.addObject('BoxROI', name='ROI', box='-0.1 -2.1 -2.1 0.1 2.1 2.1')
        # self.coarseSolution.addObject('FixedConstraint', indices='@ROI.indices')

    def onAnimateBeginEvent(self, event):
        self.start_time = process_time()

    def onAnimateEndEvent(self, event):
        self.end_time = process_time()
        print("Computation time for 1 time step: ", self.end_time - self.start_time)

def createScene(rootNode, *args, **kwargs):
    rootNode.dt = 0.005
    rootNode.gravity = [0, -12, 0]
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
