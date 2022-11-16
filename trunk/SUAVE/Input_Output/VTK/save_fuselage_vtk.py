## @ingroup Input_Output-VTK
# save_fuselage_vtk.py
#
# Created:    Jun 2021, R. Erhard
# Modified:
#

#------------------------------
# Imports
#------------------------------

from SUAVE.Input_Output.VTK.write_azimuthal_cell_values import write_azimuthal_cell_values
import numpy as np


#------------------------------
# Fuselage VTK generation
#------------------------------
## @ingroup Input_Output-VTK
def save_fuselage_vtk(vehicle, filename, Results, origin_offset):
    """
    Saves a SUAVE fuselage object as a VTK in legacy format.

    Inputs:
       vehicle        Data structure of SUAVE vehicle                [Unitless]
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
    for fuselage in vehicle.fuselages:
        fus_pts = generate_3d_fuselage_points(fuselage)
        num_fus_segs = np.shape(fus_pts)[0]
        if num_fus_segs == 0:
            print("No fuselage segments found!")
        else:
            write_fuselage_data(fus_pts,filename,origin_offset)

    return

## @ingroup Input_Output-VTK
def generate_3d_fuselage_points(fus ,tessellation = 24 ):
    """ This generates the coordinate points on the surface of the fuselage

    Assumptions:
    None

    Source:
    None

    Inputs:
    fus                  - fuselage data structure

    Properties Used:
    N/A
    """
    num_fus_segs = len(fus.Segments.keys())
    fus_pts      = np.zeros((num_fus_segs,tessellation ,3))

    if num_fus_segs > 0:
        for i_seg in range(num_fus_segs):
            theta    = np.linspace(0,2*np.pi,tessellation)
            a        = fus.Segments[i_seg].width/2
            b        = fus.Segments[i_seg].height/2
            r        = np.sqrt((b*np.sin(theta))**2  + (a*np.cos(theta))**2)
            fus_ypts = r*np.cos(theta)
            fus_zpts = r*np.sin(theta)
            fus_pts[i_seg,:,0] = fus.Segments[i_seg].percent_x_location*fus.lengths.total + fus.origin[0][0]
            fus_pts[i_seg,:,1] = fus_ypts + fus.Segments[i_seg].percent_y_location*fus.lengths.total + fus.origin[0][1]
            fus_pts[i_seg,:,2] = fus_zpts + fus.Segments[i_seg].percent_z_location*fus.lengths.total + fus.origin[0][2]

    return fus_pts

#------------------------------
# Writing fuselage data
#------------------------------
## @ingroup Input_Output-VTK
def write_fuselage_data(fus_pts,filename,origin_offset):
    """
    Writes data for a SUAVE fuselage object as a VTK in legacy format.

    Inputs:
       fus_pts        Array of nodes making up the fuselage          [Unitless]
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
                  "\nSUAVE Model of Fuselage"   ,     # Title
                  "\nASCII"                     ,     # Data type
                  "\nDATASET UNSTRUCTURED_GRID" ]     # Dataset structure / topology

        f.writelines(header)

        #---------------------
        # Write Points:
        #---------------------
        fuse_size = np.shape(fus_pts)
        n_r       = fuse_size[0]
        n_a       = fuse_size[1]
        n_indices = (n_r)*(n_a)    # total number of cell vertices
        points_header = "\n\nPOINTS "+str(n_indices) +" float"
        f.write(points_header)

        for r in range(n_r):
            for a in range(n_a):

                xp = round(fus_pts[r,a,0],4) + origin_offset[0]
                yp = round(fus_pts[r,a,1],4) + origin_offset[1]
                zp = round(fus_pts[r,a,2],4) + origin_offset[2]

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
