## @ingroup Input_Output-VTK
# save_prop_vtk.py
# 
# Created:    Jun 2021, R. Erhard
# Modified: 
#           

#----------------------------------
# Imports
#----------------------------------
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series import compute_naca_4series  
from SUAVE.Input_Output.VTK.write_azimuthal_cell_values import write_azimuthal_cell_values
from SUAVE.Core import Data
import numpy as np
import copy


def save_prop_vtk(prop, filename, Results, time_step):
    """
    Saves a SUAVE propeller object as a VTK in legacy format.

    Inputs:
       prop          Data structure of SUAVE propeller                  [Unitless] 
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
    Gprops   = generate_lofted_propeller_points(prop)
    
    for B_idx in range(n_blades):
        # Get geometry of blade for current propeller instance
        G = Gprops[B_idx]
        
        sep  = filename.find('.')
        file = filename[0:sep]+"_blade"+str(B_idx)+"_t"+str(time_step)+filename[sep:]
        
        # Create file for each blade
        with open(file, 'w') as f:
            #---------------------
            # Write header
            #--------------------- 
            header = ["# vtk DataFile Version 4.0"                ,   # File version and identifier
                      "\nSUAVE Model of PROWIM Propeller Blade "  ,   # Title 
                      "\nASCII"                                   ,   # Data type
                      "\nDATASET UNSTRUCTURED_GRID"               ]   # Dataset structure / topology     

            f.writelines(header)     
            
            # --------------------
            # Write Points
            # --------------------   
            n_r      = len(prop.chord_distribution)
            n_af     = Gprops.n_af
            
            n_vertices    = (n_r)*(n_af)    # total number of node vertices per blade
            points_header = "\n\nPOINTS "+str(n_vertices) +" float"
            f.write(points_header)
            
            # Loop over all nodes
            for r_idx in range(n_r):
                for c_idx in range(n_af):
                    xp = round(G.X[r_idx,c_idx],4)
                    yp = round(G.Y[r_idx,c_idx],4)
                    zp = round(G.Z[r_idx,c_idx],4)
                        
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

                
        f.close()
    
    
    return

def generate_lofted_propeller_points(prop):
    """
    Generates nodes on the lofted propeller.

    Inputs:
       prop          Data structure of SUAVE propeller                  [Unitless] 

    Outputs:                                   
       N/A

    Properties Used:
       N/A 
    
    Assumptions:
       Quad cell structures for mesh
    
    Source:  
       None
    
    """      
    num_B  = prop.number_of_blades      
    a_sec  = prop.airfoil_geometry          
    a_secl = prop.airfoil_polar_stations
    beta   = prop.twist_distribution         
    b      = prop.chord_distribution         
    r      = prop.radius_distribution 
    MCA    = prop.mid_chord_alignment
    t      = prop.max_thickness_distribution
    ta     = prop.orientation_euler_angles[1]
    origin = prop.origin
    
    try:
        a_o = -prop.azimuthal_offset
    except:
        # default is no azimuthal offset (blade 1 starts vertical)
        a_o = 0.0 
    
    n_a_cw    = 20                                   # number of airfoil chordwise points
    n_r       = len(b)                               # number radial points
    n_a_loft  = 2*n_a_cw                             # number points around airfoil
    theta     = np.linspace(0,2*np.pi,num_B+1)[:-1]  # azimuthal stations
    
    # create empty data structure for storing propeller geometries
    G           = Data()
    Gprops      = Data()
    Gprops.n_af = n_a_loft
     
    rot         = prop.rotation
    flip_1      = (np.pi/2)  
    flip_2      = (np.pi/2)  
    
    MCA_2d = np.repeat(np.atleast_2d(MCA).T,n_a_loft,axis=1)
    b_2d   = np.repeat(np.atleast_2d(b).T  ,n_a_loft,axis=1)
    t_2d   = np.repeat(np.atleast_2d(t).T  ,n_a_loft,axis=1)
    r_2d   = np.repeat(np.atleast_2d(r).T  ,n_a_loft,axis=1)
    
    for i in range(num_B):   
        Gprops[i] = Data()
        # get airfoil coordinate geometry   
        if a_sec != None:
            airfoil_data = import_airfoil_geometry(a_sec,npoints=n_a_cw)   
            xpts         = np.take(airfoil_data.x_coordinates,a_secl,axis=0)
            zpts         = np.take(airfoil_data.y_coordinates,a_secl,axis=0) 
            max_t        = np.take(airfoil_data.thickness_to_chord,a_secl,axis=0) 
            
        else: 
            camber       = 0.02
            camber_loc   = 0.4
            thickness    = 0.10 
            airfoil_data = compute_naca_4series(camber, camber_loc, thickness,(n_a_loft - 2))                  
            xpts         = np.repeat(np.atleast_2d(airfoil_data.x_coordinates) ,n_r,axis=0)
            zpts         = np.repeat(np.atleast_2d(airfoil_data.y_coordinates) ,n_r,axis=0)
            max_t        = np.repeat(airfoil_data.thickness_to_chord,n_r,axis=0) 
         
        # store points of airfoil in similar format as Vortex Points (i.e. in vertices)   
        max_t2d = np.repeat(np.atleast_2d(max_t).T ,n_a_loft,axis=1)
        
        xp      = rot*(- MCA_2d + xpts*b_2d)  # x coord of airfoil
        yp      = r_2d*np.ones_like(xp)       # radial location        
        zp      = zpts*(t_2d/max_t2d)         # former airfoil y coord 
                          
        matrix = np.zeros((n_r,n_a_loft,3)) # radial location, airfoil pts (same y)   
        matrix[:,:,0] = xp
        matrix[:,:,1] = yp
        matrix[:,:,2] = zp
        
        
        # ROTATION MATRICES FOR INNER SECTION     
        # rotation about y axis to create twist and position blade upright  
        trans_1 = np.zeros((n_r,3,3))
        trans_1[:,0,0] = np.cos(rot*flip_1 - rot*beta)           
        trans_1[:,0,2] = -np.sin(rot*flip_1 - rot*beta)                 
        trans_1[:,1,1] = 1
        trans_1[:,2,0] = np.sin(rot*flip_1 - rot*beta) 
        trans_1[:,2,2] = np.cos(rot*flip_1 - rot*beta) 

        # rotation about x axis to create azimuth locations 
        trans_2 = np.array([[1 , 0 , 0],
                            [0 , np.cos(theta[i] + rot*a_o + flip_2 ), -np.sin(theta[i] + rot*a_o + flip_2)],
                            [0,np.sin(theta[i] + rot*a_o + flip_2), np.cos(theta[i] + rot*a_o + flip_2)]   ]) 
        trans_2 =  np.repeat(trans_2[ np.newaxis,:,: ],n_r,axis=0)
        
        # roation about y to orient propeller/rotor to thrust angle 
        trans_3 = prop.prop_vel_to_body()
        trans_3 =  np.repeat(trans_3[ np.newaxis,:,: ],n_r,axis=0)
        
        trans   = np.matmul(trans_3,np.matmul(trans_2,trans_1))
        rot_mat = np.repeat(trans[:, np.newaxis,:,:],n_a_loft,axis=1)
         
        # ---------------------------------------------------------------------------------------------
        # ROTATE POINTS
        mat  =  np.matmul(rot_mat,matrix[...,None]).squeeze() 
        
        # ---------------------------------------------------------------------------------------------
        # store node points
        G.X  = mat[:,:,0] + origin[0][0]
        G.Y  = mat[:,:,1] + origin[0][1] 
        G.Z  = mat[:,:,2] + origin[0][2]
        
        # store cell points
        G.XA1  = mat[:-1,:-1,0] + origin[0][0]
        G.YA1  = mat[:-1,:-1,1] + origin[0][1] 
        G.ZA1  = mat[:-1,:-1,2] + origin[0][2]
        G.XA2  = mat[:-1,1:,0]  + origin[0][0]
        G.YA2  = mat[:-1,1:,1]  + origin[0][1] 
        G.ZA2  = mat[:-1,1:,2]  + origin[0][2]
                        
        G.XB1  = mat[1:,:-1,0] + origin[0][0]
        G.YB1  = mat[1:,:-1,1] + origin[0][1]  
        G.ZB1  = mat[1:,:-1,2] + origin[0][2]
        G.XB2  = mat[1:,1:,0]  + origin[0][0]
        G.YB2  = mat[1:,1:,1]  + origin[0][1]
        G.ZB2  = mat[1:,1:,2]  + origin[0][2]    
        
        # Store G for this blade:
        Gprops[i] = copy.deepcopy(G)
            
    return Gprops  
