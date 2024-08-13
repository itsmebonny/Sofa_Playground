from tracemalloc import start
from turtle import pos
from cairo import Surface
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


        filename_high = 'mesh/beam_5080.msh'
        filename_low = 'mesh/beam_410.msh'
        stl_filename = 'mesh/beam.stl'

        self.loader = rootNode.addObject('MeshSTLLoader', name='loader', filename=stl_filename)

        self.coarse = rootNode.addChild('SamplingNodes')
        self.coarse.addObject('RegularGridTopology', name='coarseGridHigh', min=p_grid.min, max=p_grid.max, nx=p_grid.res[0], ny=p_grid.res[1], nz=p_grid.res[2])
        self.coarse.addObject('TetrahedronSetTopologyContainer', name='triangleTopoHigh', src='@coarseGridHigh')
        self.MO_sampling = self.coarse.addObject('MechanicalObject', name='coarseDOFsHigh', template='Vec3d', src='@coarseGridHigh')
        self.coarse.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='1 0 0')
        #self.coarse.addObject('BarycentricMapping', name="mapping", input='@DOFs', input_topology='@triangleTopo', output='@coarseDOFsHigh', output_topology='@triangleTopoHigh')


        self.exactSolution = rootNode.addChild('HighResSolution2D', activated=True)
        self.exactSolution.addObject('MeshGmshLoader', name='grid', filename=filename_high)
        self.surface_topo = self.exactSolution.addObject('TetrahedronSetTopologyContainer', name='triangleTopo', src='@grid')
        self.MO1 = self.exactSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@grid')
        # self.exactSolution.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@quadTopo")
        self.solver = self.exactSolution.addObject('StaticSolver', name='ODE', newton_iterations="20", printLog=True)
        self.exactSolution.addObject('ParallelCGLinearSolver', template="ParallelCompressedRowSparseMatrixMat3x3d", iterations=5000, tolerance=1e-08, threshold=1e-08, warmStart=True)
        self.exactSolution.addObject('ParallelTetrahedronFEMForceField', name="FEM", youngModulus=5000, poissonRatio=0.4, method="large", updateStiffnessMatrix="false")
        self.exactSolution.addObject('BoxROI', name='ROI', box="-0.1 -1.1 -1.1 0.1 1.1 1.1")
        self.cff2 = self.exactSolution.addObject('ConstantForceField', indices="@ROI.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff_box = self.exactSolution.addObject('BoxROI', name='ROI2', box="9.9 -1.1 -1.1 10.1 1.1 1.1")
        self.cff = self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.exactSolution.addObject('BoxROI', name='ROI3', box="4.9 -1.1 -1.1 5.1 1.1 1.1")
        self.exactSolution.addObject('FixedConstraint', indices = '@ROI3.indices')
        
        self.mapping = self.exactSolution.addChild("SamplingMapping")
        self.MO_MapHR = self.mapping.addObject('MechanicalObject', name='DOFs_HR', template='Vec3d', src='@../../SamplingNodes/coarseGridHigh')
        #self.MO1_HR = self.mapping.addObject('MechanicalObject', name='DOFs_HR', template='Vec3d', position='1 3 0')
        self.mapping.addObject('BarycentricMapping', name="mapping", input='@../DOFs', input_topology='@../triangleTopo', output='@DOFs_HR')
        self.mapping.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 0')

        self.exactSolution.addChild("visual")
        self.exactSolution.visual.addObject('OglModel', src='@../../loader', color='0 1 1 0.5')
        self.exactSolution.visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')


        # same object with different resolution

        self.LowResSolution = rootNode.addChild('LowResSolution2D', activated=True)
        self.LowResSolution.addObject('MeshGmshLoader', name='gridLow', filename=filename_low)
        self.surface_topo_LR = self.LowResSolution.addObject('TetrahedronSetTopologyContainer', name='quadTopo', src='@gridLow')
        self.MO2 = self.LowResSolution.addObject('MechanicalObject', name='DOFs', template='Vec3d', src='@gridLow')
        # self.LowResSolution.addObject('MeshMatrixMass', totalMass=10, name="SparseMass", topology="@quadTopo")
        self.solver_LR = self.LowResSolution.addObject('StaticSolver', name='ODE', newton_iterations="10", printLog=True)
        self.LowResSolution.addObject('ParallelCGLinearSolver', template="ParallelCompressedRowSparseMatrixMat3x3d", iterations=2000, tolerance=1e-08, threshold=1e-08, warmStart=True) 
        self.LowResSolution.addObject('ParallelTetrahedronFEMForceField', name="FEM", youngModulus=5000, poissonRatio=0.4, method="large", updateStiffnessMatrix="false")
        self.cff_box2_LR = self.LowResSolution.addObject('BoxROI', name='ROI', box='-0.1 -1.1 -1.1 0.1 1.1 1.1')
        self.cff2LR = self.LowResSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff_box_LR = self.LowResSolution.addObject('BoxROI', name='ROI2', box="9.9 -1.1 -1.1 10.1 1.1 1.1")
        self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.LowResSolution.addObject('BoxROI', name='ROI3', box="4.9 -1.1 -1.1 5.1 1.1 1.1")
        self.LowResSolution.addObject('FixedConstraint', indices='@ROI3.indices')

        self.mapping = self.LowResSolution.addChild("SamplingMapping")
        self.MO_MapLR = self.mapping.addObject('MechanicalObject', name='DOFs_HR', template='Vec3d', src='@../../SamplingNodes/coarseGridHigh')
        #self.MO1_LR = self.mapping.addObject('MechanicalObject', name='DOFs_HR', template='Vec3d', position='1 3 0')
        self.mapping.addObject('BarycentricMapping', name="mapping", input='@../DOFs', input_topology='@../triangleTopo', output='@DOFs_HR')
        self.mapping.addObject('SphereCollisionModel', radius=sphereRadius, group=1, color='0 1 1')

        self.LowResSolution.addChild("visual")
        self.visual_model = self.LowResSolution.visual.addObject('OglModel', src='@../../loader', color='1 0 0 0.5')
        self.LowResSolution.visual.addObject('BarycentricMapping', input='@../DOFs', output='@./')

        self.nb_nodes = len(self.MO1.position.value)
        self.nb_nodes_LR = len(self.MO_MapHR.position.value)

    def onSimulationInitDoneEvent(self, event):
        """
        Called within the Sofa pipeline at the end of the scene graph initialisation.
        """
        self.inputs = []
        self.outputs = []
        self.save = True
        self.efficient_sampling = False
        if self.efficient_sampling:
            self.count_v = 0
            self.num_versors = 5
            self.versors = self.generate_versors(self.num_versors)
            self.magnitudes = np.linspace(0, 40, 30)
            self.count_m = 0
            self.angles = np.linspace(0, 2*np.pi, self.num_versors, endpoint=False)
            self.starting_points = np.linspace(self.angles[0], self.angles[1], len(self.magnitudes), endpoint=False)
        if self.save:
            if not os.path.exists('npy_gmsh'):
                os.mkdir('npy_gmsh')
            # get current time from computer format yyyy-mm-dd-hh-mm-ss and create a folder with that name
            self.directory = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            self.directory = self.directory + "_symmetric"
            if self.efficient_sampling:
                self.directory = self.directory + "_efficient"
            os.makedirs(f'npy_gmsh/{self.directory}')
            print(f"Saving data to npy_gmsh/{self.directory}")
        self.sampled = False

        surface = self.surface_topo
        surface_LR = self.surface_topo_LR

        self.idx_surface = surface.triangles.value.reshape(-1)
        self.idx_surface_LR = surface_LR.triangles.value.reshape(-1)


    def onAnimateBeginEvent(self, event):
    
        self.bad_sample = False
        # reset positions
        self.MO1.position.value = self.MO1.rest_position.value
        self.MO2.position.value = self.MO2.rest_position.value
        if self.sampled:
            print("================== Sampled all magnitudes and versors ==================\n")
            print ("================== The simulation is over ==================\n")
        
        if not self.efficient_sampling:
            self.theta = np.random.uniform(0, 2*np.pi)
            self.z = np.random.uniform(-1, 1)
            self.versor_right = np.array([np.sqrt(1 - self.z**2)*np.cos(self.theta), np.sqrt(1 - self.z**2)*np.sin(self.theta), self.z])
            self.versor_left = np.array([-np.sqrt(1 - self.z**2)*np.cos(self.theta), np.sqrt(1 - self.z**2)*np.sin(self.theta), self.z])
            self.magnitude = np.random.uniform(0, 200)
            self.externalForce_right = self.magnitude * self.versor_right
            self.externalForce_left = self.magnitude * self.versor_left
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

       
        self.exactSolution.removeObject(self.cff2)
        self.exactSolution.removeObject(self.cff)
        self.LowResSolution.removeObject(self.cff2LR)
        self.LowResSolution.removeObject(self.cffLR)

        self.cff2 = self.exactSolution.addObject('ConstantForceField', indices="@ROI.indices", totalForce=self.externalForce_left, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff = self.exactSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce_right, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff2LR = self.LowResSolution.addObject('ConstantForceField', indices="@ROI.indices", totalForce=self.externalForce_left, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cffLR = self.LowResSolution.addObject('ConstantForceField', indices="@ROI2.indices", totalForce=self.externalForce_right, showArrowSize=0.1, showColor="0.2 0.2 0.8 1")
        self.cff.init()
        self.cff2.init()
        self.cffLR.init()
        self.cff2LR.init()

        self.start_time = process_time()



    def onAnimateEndEvent(self, event):
        self.end_time = process_time()
        
        print("Computation time for 1 time step: ", self.end_time - self.start_time)
        print("External force: ", np.linalg.norm(self.externalForce_left))
        U_high = self.compute_displacement(self.MO_MapHR)
        U_low = self.compute_displacement(self.MO_MapLR)
        pos_high = self.compute_rest_position(self.MO_MapHR)
        pos_low = self.compute_rest_position(self.MO_MapLR)
        # cut the z component
        # U_high = U_high[:, :2]
        # U_low = U_low[:, :2]

        nb_nodes = self.MO_MapLR.position.value.shape[0]
        if not os.path.exists(f'npy/{nb_nodes}_nodes'):
            os.makedirs(f'npy/{nb_nodes}_nodes')
            np.save(f'npy/{nb_nodes}_nodes/HighResPoints.npy', np.array(pos_high))
            np.save(f'npy/{nb_nodes}_nodes/CoarseResPoints.npy', np.array(pos_low))

        print ("Displacement: ", np.linalg.norm(U_high - U_low))
        output = np.linalg.norm(U_high - U_low)
        self.outputs.append(output)
        if self.save and not self.bad_sample:
            if self.save and not self.efficient_sampling:    
                np.save(f'npy_gmsh/{self.directory}/HighResPoints_{round(np.linalg.norm(self.externalForce), 3)}_x_{round(self.versor_right[0], 3)}_y_{round(self.versor_right[1], 3)}.npy', np.array(U_high))
                np.save(f'npy_gmsh/{self.directory}/CoarseResPoints_{round(np.linalg.norm(self.externalForce), 3)}_x_{round(self.versor_right[0], 3)}_y_{round(self.versor_right[1], 3)}.npy', np.array(U_low))
            elif self.save and self.efficient_sampling:
                np.save(f'npy_gmsh/{self.directory}/HighResPoints_{round(self.magnitudes[self.count_m], 3)}_x_{round(self.versors[self.count_v][0], 3)}_y_{round(self.versors[self.count_v][1], 3)}.npy', np.array(U_high))
                np.save(f'npy_gmsh/{self.directory}/CoarseResPoints_{round(self.magnitudes[self.count_m], 3)}_x_{round(self.versors[self.count_v][0], 3)}_y_{round(self.versors[self.count_v][1], 3)}.npy', np.array(U_low)) 
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
    def close(self):
        if len(self.outputs) > 0:
            print("Mean error: ", np.mean(self.outputs))
            print("Max error: ", np.max(self.outputs))
            print("Min error: ", np.min(self.outputs))
            print("Standard deviation: ", np.std(self.outputs))


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
    asc = createScene(root)[1]
    Sofa.Simulation.init(root)

    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(800, 600)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()
    asc.close()


if __name__ == '__main__':
    main()
