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
    p2 = gmsh.model.geo.addPoint(5, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(7, 3, 0, lc)
    p4 = gmsh.model.geo.addPoint(9, 6, 0, lc)
    p5 = gmsh.model.geo.addPoint(10, 7, 0, lc)
    p6 = gmsh.model.geo.addPoint(7, 9, 0, lc)
    p7 = gmsh.model.geo.addPoint(2, 9, 0, lc)
    p8 = gmsh.model.geo.addPoint(-1, 8, 0, lc)
    p9 = gmsh.model.geo.addPoint(-4, 6, 0, lc)
    p10 = gmsh.model.geo.addPoint(-7, 1, 0, lc)
    p11 = gmsh.model.geo.addPoint(-5, -3, 0, lc)
    p12 = gmsh.model.geo.addPoint(-2, -4, 0, lc)

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
    l12 = gmsh.model.geo.addLine(p12, p1)

    # Create the curve loop
    ll = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12])

    # Create the surface
    s = gmsh.model.geo.addPlaneSurface([ll])

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


    mesh_type = "rectangle"


    if mesh_type == "rectangle":
        gmsh.initialize(sys.argv)
        gmsh.model.add("rectangle")
        create_rectangle(0, -1, 10, 1, 0.22)
        gmsh.finalize()
    elif mesh_type == "liver":
        gmsh.initialize(sys.argv)
        gmsh.model.add("liver")
        create_liver_like_mesh(0.25)
        gmsh.finalize()
    else:
        print("Invalid mesh type")
        sys.exit(1)


