## @ingroup Input_Output-VTK
# save_nacelle_vtk.py
#
# Created:    Jun 2021, R. Erhard
# Modified:
#

#------------------------------
# Imports
#------------------------------

from SUAVE.Input_Output.VTK.write_azimuthal_cell_values import write_azimuthal_cell_values
import numpy as np

from SUAVE.Plots.Geometry.plot_vehicle import generate_nacelle_points


#------------------------------
# Nacelle VTK generation
#------------------------------
## @ingroup Input_Output-VTK
def save_nacelle_vtk(nacelle, filename, Results, origin_offset):
    """
    Saves a SUAVE nacelle object as a VTK in legacy format.

    Inputs:
       nacelle        Data structure of SUAVE nacelle                [Unitless]
       filename       Name of vtk file to save                       [String]
       Results        Data structure of wing and propeller results   [Unitless]

    Outputs:
       N/A

    Properties Used:
       N/A

    Assumptions:
       N/A

    Source:
       None

    """

    nac_pts = generate_nacelle_points(nacelle)
    num_nac_segs = np.shape(nac_pts)[0]
    if num_nac_segs == 0:
        print("No nacelle segments found!")
    else:
        write_nacelle_data(nac_pts,filename,origin_offset)

    return


#------------------------------
# Writing nacelle data
#------------------------------
## @ingroup Input_Output-VTK
def write_nacelle_data(nac_pts,filename,origin_offset):
    """
    Writes data for a SUAVE nacelle object as a VTK in legacy format.

    Inputs:
       nac_pts        Array of nodes making up the nacelle          [Unitless]
       filename       Name of vtk file to save                       [String]

    Outputs:
       N/A

    Properties Used:
       N/A

    Assumptions:
       N/A

    Source:
       None

    """
    # Create file
    with open(filename, 'w') as f:

        #---------------------
        # Write header
        #---------------------
        header = ["# vtk DataFile Version 4.0"  ,     # File version and identifier
                  "\nSUAVE Model of nacelage"   ,     # Title
                  "\nASCII"                     ,     # Data type
                  "\nDATASET UNSTRUCTURED_GRID" ]     # Dataset structure / topology

        f.writelines(header)

        #---------------------
        # Write Points:
        #---------------------
        nac_size = np.shape(nac_pts)
        n_r       = nac_size[0]
        n_a       = nac_size[1]
        n_indices = (n_r)*(n_a)    # total number of cell vertices
        points_header = "\n\nPOINTS "+str(n_indices) +" float"
        f.write(points_header)

        for r in range(n_r):
            for a in range(n_a):

                xp = round(nac_pts[r,a,0],4) + origin_offset[0]
                yp = round(nac_pts[r,a,1],4) + origin_offset[1]
                zp = round(nac_pts[r,a,2],4) + origin_offset[2]

                new_point = "\n"+str(xp)+" "+str(yp)+" "+str(zp)
                f.write(new_point)

        #---------------------
        # Write Cells:
        #---------------------
        n            = n_a*(n_r-1) # total number of cells
        v_per_cell   = 4 # quad cells
        size         = n*(1+v_per_cell) # total number of integer values required to represent the list
        cell_header  = "\n\nCELLS "+str(n)+" "+str(size)
        f.write(cell_header)

        write_azimuthal_cell_values(f,n,n_a)

        #---------------------
        # Write Cell Types:
        #---------------------
        cell_type_header  = "\n\nCELL_TYPES "+str(n)
        f.write(cell_type_header)
        for i in range(n):
            f.write("\n9")

        #--------------------------
        # Write Scalar Cell Data:
        #--------------------------
        cell_data_header  = "\n\nCELL_DATA "+str(n)
        f.write(cell_data_header)

        # First scalar value
        f.write("\nSCALARS i float 1")
        f.write("\nLOOKUP_TABLE default")
        for i in range(n):
            new_idx = str(i)
            f.write("\n"+new_idx)


    f.close()
    return
