## @ingroup Input_Output-VTK
# save_prop_vtk.py
#
# Created:    Jun 2021, R. Erhard
# Modified:   Jul 2022, R. Erhard
#

#----------------------------------
# Imports
#----------------------------------
from MARC.Input_Output.VTK.write_azimuthal_cell_values import write_azimuthal_cell_values
from MARC.Core import Data
import numpy as np
import copy

from MARC.Visualization.Geometry.Three_Dimensional.plot_3d_rotor import get_3d_blade_coordinates

## @ingroup Input_Output-VTK
def save_prop_vtk(prop, filename, Results, time_step, origin_offset=np.array([0,0,0]), aircraftReferenceFrame=True):
    """
    Saves a MARC propeller object as a VTK in legacy format.

    Inputs:
       prop          Data structure of MARC propeller                  [Unitless]
       filename      Name of vtk file to save                           [String]
       Results       Data structure of wing and propeller results       [Unitless]
       time_step     Simulation time step                               [Unitless]

    Outputs:
       N/A

    Properties Used:
       N/A

    Assumptions:
       Quad cell structures for mesh

    Source:
       None

    """

    # Generate propeller point geometry
    n_blades = prop.number_of_blades
    n_r      = len(prop.chord_distribution)

    try:
        # Check if propeller lofted geometry has already been saved
        Gprops = prop.lofted_blade_points
        n_af     = Gprops.n_af
        # Check if there are induced velocities from wing wake at the blade (pusher config)
        velocities = prop.lofted_blade_points.induced_velocities
        # wing wake induced velocities (rotor frame)
        vt = np.reshape(velocities.vt, (n_r,n_af,n_blades))
        va = np.reshape(velocities.va, (n_r,n_af,n_blades))
        vr = np.reshape(velocities.vr, (n_r,n_af,n_blades))
        # wing wake induced velocities (wing frame)
        u = np.reshape(velocities.u, (n_r,n_af,n_blades))
        v = np.reshape(velocities.v, (n_r,n_af,n_blades))
        w = np.reshape(velocities.w, (n_r,n_af,n_blades))
        # lofted pressure distribution
        Cp = prop.lofted_blade_points.lofted_pressure_distribution
        wake=True
    except:
        # No lofted geometry has been saved yet, create it
        Gprops   = generate_lofted_propeller_points(prop,aircraftReferenceFrame)
        n_af     = prop.vtk_airfoil_points
        wake     = False

    for B_idx in range(n_blades):
        # Get geometry of blade for current propeller instance
        G = Gprops[B_idx]

        sep  = filename.rfind('.')
        file = filename[0:sep]+"_blade"+str(B_idx)+"_t."+str(time_step)+filename[sep:]

        # Create file for each blade
        with open(file, 'w') as f:
            #---------------------
            # Write header
            #---------------------
            header = ["# vtk DataFile Version 4.0"                ,   # File version and identifier
                      "\nMARC Model of PROWIM Propeller Blade "  ,   # Title
                      "\nASCII"                                   ,   # Data type
                      "\nDATASET UNSTRUCTURED_GRID"               ]   # Dataset structure / topology

            f.writelines(header)

            # --------------------
            # Write Points
            # --------------------
            n_vertices    = (n_r)*(n_af)    # total number of node vertices per blade
            points_header = "\n\nPOINTS "+str(n_vertices) +" float"
            f.write(points_header)

            # Loop over all nodes
            for r_idx in range(n_r):
                for c_idx in range(n_af):
                    xp = round(G.X[0,r_idx,c_idx],4) + origin_offset[0]
                    yp = round(G.Y[0,r_idx,c_idx],4) + origin_offset[1]
                    zp = round(G.Z[0,r_idx,c_idx],4) + origin_offset[2]

                    new_point = "\n"+str(xp)+" "+str(yp)+" "+str(zp)
                    f.write(new_point)

            #---------------------
            # Write Cells:
            #---------------------
            cells_per_blade = n_af*(n_r-1)
            v_per_cell      = 4    # quad cells
            size            = cells_per_blade*(1+v_per_cell) # total number of integer values required to represent the list
            cell_header     = "\n\nCELLS "+str(cells_per_blade)+" "+str(size)
            f.write(cell_header)

            write_azimuthal_cell_values(f, cells_per_blade, n_af)


            #---------------------
            # Write Cell Types:
            #---------------------
            cell_type_header  = "\n\nCELL_TYPES "+str(cells_per_blade)
            f.write(cell_type_header)
            for i in range(cells_per_blade):
                f.write("\n9")

            #--------------------------
            # Write Scalar Cell Data:
            #--------------------------
            cell_data_header  = "\n\nCELL_DATA "+str(cells_per_blade)
            f.write(cell_data_header)

            # First scalar value
            f.write("\nSCALARS i float 1")
            f.write("\nLOOKUP_TABLE default")
            for i in range(cells_per_blade):
                new_idx = str(i)
                f.write("\n"+new_idx)

            if wake:
                # Second scalar value
                f.write("\nSCALARS vt float")
                f.write("\nLOOKUP_TABLE default")

                for i in range(n_r-1):
                    for j in range(n_af):
                        # v shape: (n_r,n_af,n_blades)
                        # cells indexed along airfoil then radially, starting at inboard TE
                        v_i_j   = vt[i][j][B_idx]
                        v_ip_j  = vt[i+1][j][B_idx]
                        if j == n_af-1:
                            # last airfoil point connects bast to first
                            v_i_jp  = vt[i][0][B_idx]
                            v_ip_jp = vt[i+1][0][B_idx]
                        else:
                            v_i_jp  = vt[i][j+1][B_idx]
                            v_ip_jp = vt[i+1][j+1][B_idx]

                        vtavg = (v_i_j + v_ip_j + v_i_jp + v_ip_jp)/4

                        f.write("\n"+str(vtavg))

                # Third scalar value
                f.write("\nSCALARS va float")
                f.write("\nLOOKUP_TABLE default")

                for i in range(n_r-1):
                    for j in range(n_af):
                        # v shape: (n_r,n_af,n_blades)
                        # cells indexed along airfoil then radially, starting at inboard TE
                        v_i_j   = va[i][j][B_idx]
                        v_ip_j  = va[i+1][j][B_idx]
                        if j == n_af-1:
                            # last airfoil point connects bast to first
                            v_i_jp  = va[i][0][B_idx]
                            v_ip_jp = va[i+1][0][B_idx]
                        else:
                            v_i_jp  = va[i][j+1][B_idx]
                            v_ip_jp = va[i+1][j+1][B_idx]

                        vaavg = (v_i_j + v_ip_j + v_i_jp + v_ip_jp)/4

                        f.write("\n"+str(vaavg))

                # Fourth scalar value
                f.write("\nSCALARS vr float")
                f.write("\nLOOKUP_TABLE default")

                for i in range(n_r-1):
                    for j in range(n_af):
                        # v shape: (n_r,n_af,n_blades)
                        # cells indexed along airfoil then radially, starting at inboard TE
                        v_i_j   = vr[i][j][B_idx]
                        v_ip_j  = vr[i+1][j][B_idx]
                        if j == n_af-1:
                            # last airfoil point connects bast to first
                            v_i_jp  = vr[i][0][B_idx]
                            v_ip_jp = vr[i+1][0][B_idx]
                        else:
                            v_i_jp  = vr[i][j+1][B_idx]
                            v_ip_jp = vr[i+1][j+1][B_idx]

                        vravg = (v_i_j + v_ip_j + v_i_jp + v_ip_jp)/4

                        f.write("\n"+str(vravg))
                # Second scalar value
                f.write("\nSCALARS u float")
                f.write("\nLOOKUP_TABLE default")

                for i in range(n_r-1):
                    for j in range(n_af):
                        # v shape: (n_r,n_af,n_blades)
                        # cells indexed along airfoil then radially, starting at inboard TE
                        v_i_j   = u[i][j][B_idx]
                        v_ip_j  = u[i+1][j][B_idx]
                        if j == n_af-1:
                            # last airfoil point connects bast to first
                            v_i_jp  = u[i][0][B_idx]
                            v_ip_jp = u[i+1][0][B_idx]
                        else:
                            v_i_jp  = u[i][j+1][B_idx]
                            v_ip_jp = u[i+1][j+1][B_idx]

                        uavg = (v_i_j + v_ip_j + v_i_jp + v_ip_jp)/4

                        f.write("\n"+str(uavg))

                # Third scalar value
                f.write("\nSCALARS v float")
                f.write("\nLOOKUP_TABLE default")

                for i in range(n_r-1):
                    for j in range(n_af):
                        # v shape: (n_r,n_af,n_blades)
                        # cells indexed along airfoil then radially, starting at inboard TE
                        v_i_j   = v[i][j][B_idx]
                        v_ip_j  = v[i+1][j][B_idx]
                        if j == n_af-1:
                            # last airfoil point connects bast to first
                            v_i_jp  = v[i][0][B_idx]
                            v_ip_jp = v[i+1][0][B_idx]
                        else:
                            v_i_jp  = v[i][j+1][B_idx]
                            v_ip_jp = v[i+1][j+1][B_idx]

                        vavg = (v_i_j + v_ip_j + v_i_jp + v_ip_jp)/4

                        f.write("\n"+str(vavg))

                # Fourth scalar value
                f.write("\nSCALARS w float")
                f.write("\nLOOKUP_TABLE default")

                for i in range(n_r-1):
                    for j in range(n_af):
                        # v shape: (n_r,n_af,n_blades)
                        # cells indexed along airfoil then radially, starting at inboard TE
                        v_i_j   = w[i][j][B_idx]
                        v_ip_j  = w[i+1][j][B_idx]
                        if j == n_af-1:
                            # last airfoil point connects bast to first
                            v_i_jp  = w[i][0][B_idx]
                            v_ip_jp = w[i+1][0][B_idx]
                        else:
                            v_i_jp  = w[i][j+1][B_idx]
                            v_ip_jp = w[i+1][j+1][B_idx]

                        wavg = (v_i_j + v_ip_j + v_i_jp + v_ip_jp)/4

                        f.write("\n"+str(wavg))
                # Fourth scalar value
                f.write("\nSCALARS Cp float")
                f.write("\nLOOKUP_TABLE default")

                for i in range(n_r-1):
                    for j in range(n_af):
                        # v shape: (n_r,n_af,n_blades)
                        # cells indexed along airfoil then radially, starting at inboard TE
                        v_i_j   = Cp[i][j][B_idx]
                        v_ip_j  = Cp[i+1][j][B_idx]
                        if j == n_af-1:
                            # last airfoil point connects bast to first
                            v_i_jp  = Cp[i][0][B_idx]
                            v_ip_jp = Cp[i+1][0][B_idx]
                        else:
                            v_i_jp  = Cp[i][j+1][B_idx]
                            v_ip_jp = Cp[i+1][j+1][B_idx]

                        Cp_avg = (v_i_j + v_ip_j + v_i_jp + v_ip_jp)/4

                        f.write("\n"+str(Cp_avg))
        f.close()


    return

## @ingroup Input_Output-VTK
def generate_lofted_propeller_points(prop,aircraftReferenceFrame):
    """
    Generates nodes on the lofted propeller in the propeller frame.

    Inputs:
       prop          Data structure of MARC propeller                  [Unitless]

    Outputs:
       N/A

    Properties Used:
       N/A

    Assumptions:
       Quad cell structures for mesh

    Source:
       None

    """
    # unpack
    num_B    = prop.number_of_blades
    n_points = prop.vtk_airfoil_points
    dim      = len(prop.radius_distribution)
    
    # Initialize data structure for propellers
    Gprops = Data()
    
    for i in range(num_B):
        Gprops[i] = Data()
        G = get_3d_blade_coordinates(prop,n_points,dim,i,aircraftRefFrame=aircraftReferenceFrame)

        # Store G for this blade:
        Gprops[i] = copy.deepcopy(G)


    return Gprops
