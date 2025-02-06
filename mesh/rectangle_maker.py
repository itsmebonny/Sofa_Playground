import gmsh
import sys

# Create a rectangle in the grid defined by the points (x_min, y_min) and (x_max, y_max)

def create_rectangle(x_min, y_min, x_max, y_max, lc):
    """
    Create a rectangle in the grid defined by the points (x_min, y_min) and (x_max, y_max)
    """
    p1 = gmsh.model.geo.addPoint(x_min, y_min, 0, lc)
    p2 = gmsh.model.geo.addPoint(x_max, y_min, 0, lc)
    p3 = gmsh.model.geo.addPoint(x_max, y_max, 0, lc)
    p4 = gmsh.model.geo.addPoint(x_min, y_max, 0, lc)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    ll = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([ll])

    # Create a physical group for the rectangle
    gmsh.model.addPhysicalGroup(2, [s], 1)
    gmsh.model.setPhysicalName(2, 1, "rectangle")

    # generate the mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    #compute the number of nodes
    nodes = gmsh.model.mesh.getNodes()
    nb_nodes = len(nodes[0])

    # Save the mesh

    gmsh.write(f"mesh/rectangle_{nb_nodes}.msh")

    # Launch the GUI to see the results
    gmsh.fltk.run()


import numpy as np
from stl import mesh

def create_beam_stl(x_min, y_min, x_max, y_max, z_min, z_max, filename):
    # Define the 8 vertices of the beam
    vertices = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max]
    ])
    
    # Define the 12 triangles composing the beam
    faces = np.array([
        [0, 3, 1], [1, 3, 2],  # Bottom
        [0, 1, 4], [1, 5, 4],  # Front
        [1, 2, 5], [2, 6, 5],  # Right
        [2, 3, 6], [3, 7, 6],  # Back
        [3, 0, 7], [0, 4, 7],  # Left
        [4, 5, 6], [4, 6, 7]   # Top
    ])
    
    # Create the mesh
    beam_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            beam_mesh.vectors[i][j] = vertices[face[j], :]
    
    # Write the mesh to file
    beam_mesh.save(filename)



def create_beam(x_min, y_min, x_max, y_max, z_min, z_max, lc):
    # Create the points
    p1 = gmsh.model.geo.addPoint(x_min, y_min, z_min, lc)
    p2 = gmsh.model.geo.addPoint(x_max, y_min, z_min, lc)
    p3 = gmsh.model.geo.addPoint(x_max, y_max, z_min, lc)
    p4 = gmsh.model.geo.addPoint(x_min, y_max, z_min, lc)
    p5 = gmsh.model.geo.addPoint(x_min, y_min, z_max, lc)
    p6 = gmsh.model.geo.addPoint(x_max, y_min, z_max, lc)
    p7 = gmsh.model.geo.addPoint(x_max, y_max, z_max, lc)
    p8 = gmsh.model.geo.addPoint(x_min, y_max, z_max, lc)

    # Create the lines
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8 = gmsh.model.geo.addLine(p8, p5)
    l9 = gmsh.model.geo.addLine(p1, p5)
    l10 = gmsh.model.geo.addLine(p2, p6)
    l11 = gmsh.model.geo.addLine(p3, p7)
    l12 = gmsh.model.geo.addLine(p4, p8)

    # Create the curve loop
    cl1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    cl2 = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
    cl3 = gmsh.model.geo.addCurveLoop([-l9, l1, l10, -l5])
    cl4 = gmsh.model.geo.addCurveLoop([-l3, l11, l7, -l12])
    cl5 = gmsh.model.geo.addCurveLoop([l10, l6, -l11, -l2])
    cl6 = gmsh.model.geo.addCurveLoop([l9, -l8, -l12, l4])

    # Create the surface
    s1 = gmsh.model.geo.addPlaneSurface([cl1])
    s2 = gmsh.model.geo.addPlaneSurface([cl2])
    s3 = gmsh.model.geo.addPlaneSurface([cl3])
    s4 = gmsh.model.geo.addPlaneSurface([cl4])
    s5 = gmsh.model.geo.addPlaneSurface([cl5])
    s6 = gmsh.model.geo.addPlaneSurface([cl6])

    # create the volume
    v = gmsh.model.geo.addSurfaceLoop([s1, s2, s3, s4, s5, s6])
    v = gmsh.model.geo.addVolume([v])

    # Create a physical group for the beam
    gmsh.model.addPhysicalGroup(3, [v], 1)
    gmsh.model.setPhysicalName(3, 1, "beam")

    # generate the mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    #compute the number of nodes
    nodes = gmsh.model.mesh.getNodes()
    nb_nodes = len(nodes[0])

    # Save the mesh
    gmsh.write(f"mesh/beam_{nb_nodes}.msh")



    # Launch the GUI to see the results
    gmsh.fltk.run()

def create_liver_like_mesh(lc):
    """
    Create a liver like mesh
    """
    # Create the points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(2, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(5, 1, 0, lc)
    p4 = gmsh.model.geo.addPoint(7, 3, 0, lc)
    p5 = gmsh.model.geo.addPoint(9, 6, 0, lc)
    p6 = gmsh.model.geo.addPoint(11, 8, 0, lc)
    p7 = gmsh.model.geo.addPoint(10, 9, 0, lc)
    p8 = gmsh.model.geo.addPoint(7, 10, 0, lc)
    p9 = gmsh.model.geo.addPoint(4, 10, 0, lc)
    p10 = gmsh.model.geo.addPoint(2, 9.6, 0, lc)
    p11 = gmsh.model.geo.addPoint(0, 10, 0, lc)
    p12 = gmsh.model.geo.addPoint(-3, 10, 0, lc)
    p13 = gmsh.model.geo.addPoint(-5, 9, 0, lc)
    p14 = gmsh.model.geo.addPoint(-7, 8, 0, lc)
    p15 = gmsh.model.geo.addPoint(-8, 6, 0, lc)
    p16 = gmsh.model.geo.addPoint(-8.5, 3, 0, lc)
    p17 = gmsh.model.geo.addPoint(-9, -1, 0, lc)
    p18 = gmsh.model.geo.addPoint(-9.5, -3.5, 0, lc)
    p19 = gmsh.model.geo.addPoint(-6.5, -3.5, 0, lc)
    p20 = gmsh.model.geo.addPoint(-5, -3, 0, lc)
    p21 = gmsh.model.geo.addPoint(-3, -2, 0, lc)
    p22 = gmsh.model.geo.addPoint(-1, -1, 0, lc)

    # Create the lines
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p5)
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8 = gmsh.model.geo.addLine(p8, p9)
    l9 = gmsh.model.geo.addLine(p9, p10)
    l10 = gmsh.model.geo.addLine(p10, p11)
    l11 = gmsh.model.geo.addLine(p11, p12)
    l12 = gmsh.model.geo.addLine(p12, p13)
    l13 = gmsh.model.geo.addLine(p13, p14)
    l14 = gmsh.model.geo.addLine(p14, p15)
    l15 = gmsh.model.geo.addLine(p15, p16)
    l16 = gmsh.model.geo.addLine(p16, p17)
    l17 = gmsh.model.geo.addLine(p17, p18)
    l18 = gmsh.model.geo.addLine(p18, p19)
    l19 = gmsh.model.geo.addLine(p19, p20)
    l20 = gmsh.model.geo.addLine(p20, p21)
    l21 = gmsh.model.geo.addLine(p21, p22)
    l22 = gmsh.model.geo.addLine(p22, p1)

    # Create the curve loop
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11,
                                      l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22])
    
    # Create the surface
    s = gmsh.model.geo.addPlaneSurface([cl])

    # Create a physical group for the liver
    gmsh.model.addPhysicalGroup(2, [s], 1)
    gmsh.model.setPhysicalName(2, 1, "liver")

    # generate the mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    #compute the number of nodes
    nodes = gmsh.model.mesh.getNodes()
    nb_nodes = len(nodes[0])

    # Save the mesh
    gmsh.write(f"mesh/liver_{nb_nodes}.msh")

    # Launch the GUI to see the results
    gmsh.fltk.run()

import gmsh

def create_liver_like_mesh_3D(lc, depth=0.1):
    gmsh.initialize()

    gmsh.model.add("Liver")

    # Create the points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(2, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(5, 1, 0, lc)
    p4 = gmsh.model.geo.addPoint(7, 3, 0, lc)
    p5 = gmsh.model.geo.addPoint(9, 6, 0, lc)
    p6 = gmsh.model.geo.addPoint(11, 8, 0, lc)
    p7 = gmsh.model.geo.addPoint(10, 9, 0, lc)
    p8 = gmsh.model.geo.addPoint(7, 10, 0, lc)
    p9 = gmsh.model.geo.addPoint(4, 10, 0, lc)
    p10 = gmsh.model.geo.addPoint(2, 9.6, 0, lc)
    p11 = gmsh.model.geo.addPoint(0, 10, 0, lc)
    p12 = gmsh.model.geo.addPoint(-3, 10, 0, lc)
    p13 = gmsh.model.geo.addPoint(-5, 9, 0, lc)
    p14 = gmsh.model.geo.addPoint(-7, 8, 0, lc)
    p15 = gmsh.model.geo.addPoint(-8, 6, 0, lc)
    p16 = gmsh.model.geo.addPoint(-8.5, 3, 0, lc)
    p17 = gmsh.model.geo.addPoint(-9, -1, 0, lc)
    p18 = gmsh.model.geo.addPoint(-9.5, -3.5, 0, lc)
    p19 = gmsh.model.geo.addPoint(-6.5, -3.5, 0, lc)
    p20 = gmsh.model.geo.addPoint(-5, -3, 0, lc)
    p21 = gmsh.model.geo.addPoint(-3, -2, 0, lc)
    p22 = gmsh.model.geo.addPoint(-1, -1, 0, lc)

    # Create the lines
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p5)
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8 = gmsh.model.geo.addLine(p8, p9)
    l9 = gmsh.model.geo.addLine(p9, p10)
    l10 = gmsh.model.geo.addLine(p10, p11)
    l11 = gmsh.model.geo.addLine(p11, p12)
    l12 = gmsh.model.geo.addLine(p12, p13)
    l13 = gmsh.model.geo.addLine(p13, p14)
    l14 = gmsh.model.geo.addLine(p14, p15)
    l15 = gmsh.model.geo.addLine(p15, p16)
    l16 = gmsh.model.geo.addLine(p16, p17)
    l17 = gmsh.model.geo.addLine(p17, p18)
    l18 = gmsh.model.geo.addLine(p18, p19)
    l19 = gmsh.model.geo.addLine(p19, p20)
    l20 = gmsh.model.geo.addLine(p20, p21)
    l21 = gmsh.model.geo.addLine(p21, p22)
    l22 = gmsh.model.geo.addLine(p22, p1)

    # Create the curve loop
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11,
                                      l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22])

    # Create the surface
    s = gmsh.model.geo.addPlaneSurface([cl])

    

    # Extrude the surface to add depth
    gmsh.model.geo.extrude([(2, s)], 10, 10, depth, [2])

    gmsh.model.geo.synchronize()

    # Generate the mesh
    gmsh.model.mesh.generate(3)

    nodes = gmsh.model.mesh.getNodes()
    nb_nodes = len(nodes[0])

    # Save the mesh
    gmsh.write(f"mesh/liver_{nb_nodes}.msh")


    # Launch the GUI to see the results
    gmsh.fltk.run()



def mesh_stl_with_char_length(stl_file, max_char_length):
    import math
    gmsh.initialize()

    # Add a new model
    gmsh.model.add("STL to Mesh")

    # Import the STL file
    gmsh.merge(stl_file)
    # angle = 40

    # # For complex geometries, patches can be too complex, too elongated or too
    # # large to be parametrized; setting the following option will force the
    # # creation of patches that are amenable to reparametrization:
    # forceParametrizablePatches = 0

    # # For open surfaces include the boundary edges in the classification
    # # process:
    # includeBoundary = True

    # # Force curves to be split on given angle:
    # curveAngle = 90

    # gmsh.model.mesh.classifySurfaces(angle * math.pi / 180., includeBoundary,
    #                                  forceParametrizablePatches,
    #                                  curveAngle * math.pi / 180.)

    # gmsh.model.mesh.createGeometry()
    n = gmsh.model.getDimension()
    s = gmsh.model.getEntities(n)
    l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
    gmsh.model.geo.addVolume([l])
    print("Volume added")
    gmsh.model.geo.synchronize()

    # Set characteristic length
    # gmsh.option.setNumber("Geometry.Tolerance", 1e-3)
    # gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.1)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # 1=Delaunay, 4=Frontal, 7=MMG3D, 10=HXT
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_char_length)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 5)
    
    # Generate 3D mesh
    gmsh.model.mesh.generate(3)
    nodes = gmsh.model.mesh.getNodes()
    nb_nodes = len(nodes[0])


    # Save the mesh to a file
    gmsh.write(f"mesh/liver_{nb_nodes}.msh")
    

    gmsh.fltk.run()


def create_high_freq_2D(zigzag_amplitude=0.7, x_min=0, y_min=-1, x_max=10, y_max=1, 
                       n_zigzags_horizontal=160, n_zigzags_vertical=30, lc=0.5):
    """
    Create a rectangle with zigzag edges with different numbers of horizontal and vertical zigzags
    zigzag_amplitude: amplitude of the zigzag
    n_zigzags_horizontal: number of zigzags on top and bottom edges
    n_zigzags_vertical: number of zigzags on left and right edges 
    """
    points = []
    lines = []
    
    # Bottom edge - zigzags pointing outwards
    x_step = (x_max - x_min) / n_zigzags_horizontal
    for i in range(n_zigzags_horizontal + 1):
        x = x_min + i * x_step
        y = y_min + (zigzag_amplitude if i % 2 else 0)  # Changed + to - to point outwards
        points.append(gmsh.model.geo.addPoint(x, y, 0, lc))
    
    # Right edge
    y_step = (y_max - y_min) / n_zigzags_vertical
    for i in range(1, n_zigzags_vertical + 1):
        y = y_min + i * y_step
        x = x_max #+ (zigzag_amplitude if i % 2 else 0)
        points.append(gmsh.model.geo.addPoint(x, y, 0, lc))
    
    # Top edge
    for i in range(1, n_zigzags_horizontal + 1):
        x = x_max - i * x_step
        y = y_max - (zigzag_amplitude if i % 2 else 0)
        points.append(gmsh.model.geo.addPoint(x, y, 0, lc))
    
    # Left edge (straight line)
    y_step = (y_max - y_min) / n_zigzags_vertical
    for i in range(1, n_zigzags_vertical):
        y = y_max - i * y_step
        x = x_min
        points.append(gmsh.model.geo.addPoint(x, y, 0, lc))
    
    # Create lines connecting all points
    for i in range(len(points)):
        lines.append(gmsh.model.geo.addLine(points[i], points[(i + 1) % len(points)]))
    
    # Create curve loop and surface
    cl = gmsh.model.geo.addCurveLoop(lines)
    s = gmsh.model.geo.addPlaneSurface([cl])
    
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    
    nodes = gmsh.model.mesh.getNodes()
    nb_nodes = len(nodes[0])
    
    gmsh.write(f"mesh/zigzag_{nb_nodes}.msh")
    gmsh.fltk.run()





def create_lego_brick(length=10, width=5, height=1, stud_height=0.5, 
                     n_studs_x=4, n_studs_y=2, stud_size=0.5, lc=0.2):
    gmsh.initialize()
    gmsh.model.add("lego_brick")
    
    # Switch to OpenCASCADE CAD kernel
    gmsh.option.setNumber("Geometry.OCCAutoFix", 1)
    
    # Create base brick
    box = gmsh.model.occ.addBox(0, 0, 0, length, width, height)
    
    # Calculate stud spacing
    x_margin = (length - (n_studs_x * stud_size)) / (n_studs_x + 1)
    y_margin = (width - (n_studs_y * stud_size)) / (n_studs_y + 1)
    
    # Create studs using OpenCASCADE
    studs = []
    for i in range(n_studs_x):
        for j in range(n_studs_y):
            x_start = x_margin + (i * (stud_size + x_margin))
            y_start = y_margin + (j * (stud_size + y_margin))
            stud = gmsh.model.occ.addBox(x_start, y_start, height, 
                                       stud_size, stud_size, stud_height)
            studs.append(stud)
    
    # Fuse all volumes
    if studs:
        out, _ = gmsh.model.occ.fuse([(3, box)], [(3, s) for s in studs])
    
    # Synchronize before meshing
    gmsh.model.occ.synchronize()
    
    # Mesh settings
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    
    # Generate full 3D mesh once
    gmsh.model.mesh.generate(3)
    
    # Save volume mesh
    gmsh.write(f"mesh/lego_brick.msh")
    nodes = gmsh.model.mesh.getNodes()
    nb_nodes = len(nodes[0])
    gmsh.write(f"mesh/lego_brick_{nb_nodes}.msh")
    
    # Save surface mesh by getting boundary
    # Get all surface elements without regenerating mesh
    surface_dimTags = gmsh.model.getBoundary([(3, box)], combined=False, oriented=True)
    
    # Create a new physical group for surfaces
    surface_group = gmsh.model.addPhysicalGroup(2, [tag for dim, tag in surface_dimTags])
    
    # Save only the surface elements
    gmsh.option.setNumber("Mesh.SaveAll", 0)  # Only save physical groups
    gmsh.write(f"mesh/lego_for_collision.msh")
    
    # Show result
    gmsh.fltk.run()
    gmsh.finalize()


def create_plate_with_hole(length=10, width=5, thickness=1,
                          hole_radius=1, hole_x=5, hole_y=2.5, lc=0.5):
    """Create 3D plate with circular hole"""
    
    # Create plate outline
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(length, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(length, width, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, width, 0, lc)

    # Create edges
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # Create hole - center point and 3 points on circle
    pc = gmsh.model.geo.addPoint(hole_x, hole_y, 0, lc)
    p5 = gmsh.model.geo.addPoint(hole_x + hole_radius, hole_y, 0, lc)
    p6 = gmsh.model.geo.addPoint(hole_x, hole_y + hole_radius, 0, lc)
    p7 = gmsh.model.geo.addPoint(hole_x - hole_radius, hole_y, 0, lc)
    p8 = gmsh.model.geo.addPoint(hole_x, hole_y - hole_radius, 0, lc)

    # Create circle arcs
    c1 = gmsh.model.geo.addCircleArc(p5, pc, p6)
    c2 = gmsh.model.geo.addCircleArc(p6, pc, p7)
    c3 = gmsh.model.geo.addCircleArc(p7, pc, p8)
    c4 = gmsh.model.geo.addCircleArc(p8, pc, p5)

    # Create loops and surface
    plate_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    hole_loop = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
    surface = gmsh.model.geo.addPlaneSurface([plate_loop, hole_loop])

    # Create volume
    volume = gmsh.model.geo.extrude([(2, surface)], 0, 0, thickness)
    
    # Generate mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    nodes = gmsh.model.mesh.getNodes()
    nb_nodes = len(nodes[0])
    
    # Save mesh with parameters in filename
    filename = f"plate_x{hole_x}_y{hole_y}_r{hole_radius}_n{nb_nodes}.msh"
    directory = "mesh/"
    gmsh.write(directory + filename)
    
    
    # Launch GUI
    gmsh.fltk.run()


import numpy as np
if __name__ == '__main__':


    mesh_type = "plate"
    input_stl_file = "mesh/liver_lowres.stl"


    if mesh_type == "rectangle":
        gmsh.initialize(sys.argv)
        gmsh.model.add("rectangle")
        create_rectangle(0, -1, 10, 1, 0.22)
        gmsh.finalize()
    elif mesh_type == "liver":
        gmsh.initialize(sys.argv)
        gmsh.model.add("liver")
        for i in np.arange(1, 3, 0.1):
            create_liver_like_mesh_3D(i, 0.5)
        gmsh.finalize()
    elif mesh_type == "beam":
        
        for i in np.arange(0.4, 0.6, 0.02):
            gmsh.initialize(sys.argv)
            gmsh.model.add("beam")
            create_beam(0, -1, 10, 1, -1, 1, i)
            gmsh.finalize()
    elif mesh_type == "stl":
        for i in np.arange(0.4, 0.6, 0.02):
            mesh_stl_with_char_length(input_stl_file, i)
            gmsh.finalize()
            print("STL mode")
    elif mesh_type == "zigzag":
        gmsh.initialize(sys.argv)
        gmsh.model.add("zigzag")
        create_high_freq_2D()
        gmsh.finalize()
    elif mesh_type == "lego":
        create_lego_brick(length=10, width=5, height=1, stud_height=5,
                     n_studs_x=6, n_studs_y=3, stud_size=1, lc=0.5)
        
        # Add to main section:
    elif mesh_type == "plate":
        n_simulations = 15
        length = 10
        width = 5
        thickness = 1
        for i in range(n_simulations):
            hole_x = np.random.randint(2, 8)
            hole_y = np.random.randint(1, 4)
            hole_radius = round(np.random.uniform(0.5, 1.5), 2)
            #check if the hole is not too close to the edge
            while hole_x - hole_radius < 0.5 or hole_x + hole_radius > 9.5 or hole_y - hole_radius < 0.5 or hole_y + hole_radius > 4.5:
                hole_x = np.random.randint(2, 8)
                hole_y = np.random.randint(1, 4)
                hole_radius = round(np.random.uniform(0.5, 1.5), 2)
            gmsh.initialize(sys.argv)
            gmsh.model.add("plate")
            create_plate_with_hole(length, width, thickness, hole_radius, hole_x, hole_y, 0.5)
            gmsh.finalize()
            gmsh.initialize(sys.argv)
            gmsh.model.add("plate")
            create_plate_with_hole(length, width, thickness, hole_radius, hole_x, hole_y, 2)
            gmsh.finalize()
