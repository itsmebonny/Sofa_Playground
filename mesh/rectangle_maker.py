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
    


    
    
    

if __name__ == '__main__':


    mesh_type = "liver"


    if mesh_type == "rectangle":
        gmsh.initialize(sys.argv)
        gmsh.model.add("rectangle")
        create_rectangle(0, -1, 10, 1, 0.22)
        gmsh.finalize()
    elif mesh_type == "liver":
        gmsh.initialize(sys.argv)
        gmsh.model.add("liver")
        create_liver_like_mesh(1.3)
        gmsh.finalize()
    else:
        print("Invalid mesh type")
        sys.exit(1)


