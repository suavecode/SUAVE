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


#------------------------------
# Nacelle VTK generation
#------------------------------
## @ingroup Input_Output-VTK
def save_nacelle_vtk(nacelle, filename, Results):
    """
    Saves a SUAVE nacelle object as a VTK in legacy format.

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

    nac_pts = generate_nacelle_points(nacelle)
    num_nac_segs = np.shape(nac_pts)[0]
    if num_nac_segs == 0:
        print("No nacelle segments found!")
    else:
        write_nacelle_data(nac_pts,filename)

    return

## @ingroup Input_Output-VTK
def generate_nacelle_points(nac,tessellation = 24):
    """ This generates the coordinate points on the surface of the nacelle

    Assumptions:
    None

    Source:
    None

    Inputs: 
    Properties Used:
    N/A 
    """
     
    
    num_nac_segs = len(nac.Segments.keys())   
    theta        = np.linspace(0,2*np.pi,tessellation)
    n_points     = 20
    
    if num_nac_segs == 0:
        num_nac_segs = int(n_points/2)
        nac_pts      = np.zeros((num_nac_segs,tessellation,3))
        naf          = nac.Airfoil
        
        if naf.naca_4_series_airfoil != None: 
            # use mean camber surface of airfoil
            camber       = float(naf.naca_4_series_airfoil[0])/100
            camber_loc   = float(naf.naca_4_series_airfoil[1])/10
            thickness    = float(naf.naca_4_series_airfoil[2:])/100 
            airfoil_data = compute_naca_4series(camber, camber_loc, thickness,(n_points - 2))
            xpts         = np.repeat(np.atleast_2d(airfoil_data.x_lower_surface).T,tessellation,axis = 1)*nac.length 
            zpts         = np.repeat(np.atleast_2d(airfoil_data.camber_coordinates[0]).T,tessellation,axis = 1)*nac.length  
        
        elif naf.coordinate_file != None: 
            a_sec        = naf.coordinate_file
            a_secl       = [0]
            airfoil_data = import_airfoil_geometry(a_sec,npoints=num_nac_segs)
            xpts         = np.repeat(np.atleast_2d(np.take(airfoil_data.x_coordinates,a_secl,axis=0)).T,tessellation,axis = 1)*nac.length  
            zpts         = np.repeat(np.atleast_2d(np.take(airfoil_data.y_coordinates,a_secl,axis=0)).T,tessellation,axis = 1)*nac.length  
        
        else:
            # if no airfoil defined, use super ellipse as default
            a =  nac.length/2 
            b =  (nac.diameter - nac.inlet_diameter)/2 
            b = np.maximum(b,1E-3) # ensure 
            xpts =  np.repeat(np.atleast_2d(np.linspace(-a,a,num_nac_segs)).T,tessellation,axis = 1) 
            zpts = (np.sqrt((b**2)*(1 - (xpts**2)/(a**2) )))*nac.length 
            xpts = (xpts+a)*nac.length  

        if nac.flow_through: 
            zpts = zpts + nac.inlet_diameter/2  
                
        # create geometry 
        theta_2d = np.repeat(np.atleast_2d(theta),num_nac_segs,axis =0) 
        nac_pts[:,:,0] =  xpts
        nac_pts[:,:,1] =  zpts*np.cos(theta_2d)
        nac_pts[:,:,2] =  zpts*np.sin(theta_2d)  
                
    else:
        nac_pts = np.zeros((num_nac_segs,tessellation,3)) 
        for i_seg in range(num_nac_segs):
            a        = nac.Segments[i_seg].width/2
            b        = nac.Segments[i_seg].height/2
            r        = np.sqrt((b*np.sin(theta))**2  + (a*np.cos(theta))**2)
            nac_ypts = r*np.cos(theta)
            nac_zpts = r*np.sin(theta)
            nac_pts[i_seg,:,0] = nac.Segments[i_seg].percent_x_location*nac.length
            nac_pts[i_seg,:,1] = nac_ypts + nac.Segments[i_seg].percent_y_location*nac.length 
            nac_pts[i_seg,:,2] = nac_zpts + nac.Segments[i_seg].percent_z_location*nac.length  
            
    # rotation about y to orient propeller/rotor to thrust angle
    rot_trans =  nac.nac_vel_to_body()
    rot_trans =  np.repeat( np.repeat(rot_trans[ np.newaxis,:,: ],tessellation,axis=0)[ np.newaxis,:,:,: ],num_nac_segs,axis=0)    
    
    NAC_PTS  =  np.matmul(rot_trans,nac_pts[...,None]).squeeze()  
     
    # translate to body 
    NAC_PTS[:,:,0] = NAC_PTS[:,:,0] + nac.origin[0][0]
    NAC_PTS[:,:,1] = NAC_PTS[:,:,1] + nac.origin[0][1]
    NAC_PTS[:,:,2] = NAC_PTS[:,:,2] + nac.origin[0][2]
    return NAC_PTS




#------------------------------
# Writing nacelle data
#------------------------------
## @ingroup Input_Output-VTK
def write_nacelle_data(nac_pts,filename):
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

                xp = round(nac_pts[r,a,0],4)
                yp = round(nac_pts[r,a,1],4)
                zp = round(nac_pts[r,a,2],4)

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
