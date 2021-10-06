## @ingroup Input_Output-VTK
# save_vortex_distribution_vtk.py
# 
# Created:    Jun 2021, R. Erhard
# Modified: 
#           
from SUAVE.Core import Data
import numpy as np

def save_vortex_distribution_vtk(vehicle,conditions,VD,gamma,wing_instance,filename, time_step,separate_wing_and_wake_VD=True):
    """
    Saves a SUAVE propeller wake as a VTK in legacy format.

    Inputs:
       VD           Vortex distribution of propeller wake          [Unitless]  
       filename      Name of vtk file to save                       [Unitless]  
       Results       Data structure of wing and propeller results   [Unitless]  
       i_prop        ith propeller to evaluate wake of              [Unitless]
       
    Outputs:                                   
       N/A

    Properties Used:
       N/A 
    
    Assumptions:
       Quad cell structures for mesh
    
    Source:  
       None
    
    """
    # Extract wake properties of the ith propeller
    symmetric = vehicle.wings[wing_instance.tag].symmetric
    alpha = -conditions.aerodynamics.angle_of_attack[0][0] # rotating back towards freestream
    if symmetric:
        # generate VLM horseshoes for left and right wings
        half_l = int(len(VD.XA1)/2)
        
        L = Data()
        R = Data()
        L.XAH = VD.XAH[half_l:]
        L.YAH = VD.YAH[half_l:]
        L.ZAH = VD.ZAH[half_l:]
        L.XBH = VD.XBH[half_l:]
        L.YBH = VD.YBH[half_l:]
        L.ZBH = VD.ZBH[half_l:]
        L.XA2 = VD.XA2[half_l:]
        L.YA2 = VD.YA2[half_l:]
        L.ZA2 = VD.ZA2[half_l:]
        L.XB2 = VD.XB2[half_l:]
        L.YB2 = VD.YB2[half_l:]
        L.ZB2 = VD.ZB2[half_l:]
        L.XB_TE = VD.XB_TE[half_l:]
        L.YB_TE = VD.YB_TE[half_l:]
        L.ZB_TE = VD.ZB_TE[half_l:]
        L.XA_TE = VD.XA_TE[half_l:]
        L.YA_TE = VD.YA_TE[half_l:]
        L.ZA_TE = VD.ZA_TE[half_l:]  
        L.gamma = gamma[0][half_l:] 
        
        R.XAH = VD.XAH[0:half_l]
        R.YAH = VD.YAH[0:half_l]
        R.ZAH = VD.ZAH[0:half_l]
        R.XBH = VD.XBH[0:half_l]
        R.YBH = VD.YBH[0:half_l]
        R.ZBH = VD.ZBH[0:half_l]
        R.XA2 = VD.XA2[0:half_l]
        R.YA2 = VD.YA2[0:half_l]
        R.ZA2 = VD.ZA2[0:half_l]
        R.XB2 = VD.XB2[0:half_l]
        R.YB2 = VD.YB2[0:half_l]
        R.ZB2 = VD.ZB2[0:half_l]
        R.XB_TE = VD.XB_TE[0:half_l]
        R.YB_TE = VD.YB_TE[0:half_l]
        R.ZB_TE = VD.ZB_TE[0:half_l]
        R.XA_TE = VD.XA_TE[0:half_l]
        R.YA_TE = VD.YA_TE[0:half_l]
        R.ZA_TE = VD.ZA_TE[0:half_l]   
        R.gamma = gamma[0][0:half_l]     
        
        sep  = filename.find('.')
        
        Lfile = filename[0:sep]+"_L"+filename[sep:]
        Rfile = filename[0:sep]+"_R"+filename[sep:]
                
        write_vortex_distribution_vtk(R,alpha,VD,Rfile,time_step,separate_wing_and_wake_VD)
        write_vortex_distribution_vtk(L,alpha,VD,Lfile,time_step,separate_wing_and_wake_VD)
    else:
        wing = Data()
        wing.XAH = VD.XAH
        wing.YAH = VD.YAH
        wing.ZAH = VD.ZAH
        wing.XBH = VD.XBH
        wing.YBH = VD.YBH
        wing.ZBH = VD.ZBH
        wing.XA2 = VD.XA2
        wing.YA2 = VD.YA2
        wing.ZA2 = VD.ZA2
        wing.XB2 = VD.XB2
        wing.YB2 = VD.YB2
        wing.ZB2 = VD.ZB2
        wing.XB_TE = VD.XB_TE
        wing.YB_TE = VD.YB_TE
        wing.ZB_TE = VD.ZB_TE  
        wing.XA_TE = VD.XA_TE
        wing.YA_TE = VD.YA_TE
        wing.ZA_TE = VD.ZA_TE    
        wing.gamma = gamma[0]

        file = filename
        
        write_vortex_distribution_vtk(wing,alpha,VD,file,time_step,separate_wing_and_wake_VD)
        
        
    return

def write_vortex_distribution_vtk(wing,alpha,VD,filename,time_step, separate_wing_and_wake_VD=True):
 
    
    n_sw = VD.n_sw[0]
    n_cw = VD.n_cw[0]
    n_cp = n_sw*n_cw
    
    # Reformat to obtain vortex data structures
    right_trailing_vortices, bound_vortices, inf_trailing_vortices = process_VD(wing, n_sw, n_cw)    
    
    sep  = filename.find('.')

    wingVDfile = filename[0:sep]+"_wingVD."+str(time_step)+filename[sep:]   
    wakeVDfile = filename[0:sep]+"_wakeVD."+str(time_step)+filename[sep:] 
    fullVDfile = filename[0:sep]+str(time_step)+filename[sep:]        
    
    if separate_wing_and_wake_VD:
        # Write vortex distribution on wing panels
        wing_VD(wingVDfile,n_sw,n_cp,bound_vortices, right_trailing_vortices)
        
        # Write vortex distribution of infinite trailing vortices
        wake_VD(wakeVDfile,n_cp,n_sw,inf_trailing_vortices)
    else:
        full_VD(fullVDfile, n_cp, n_sw, bound_vortices, right_trailing_vortices, inf_trailing_vortices)
    
    

    return


def process_VD(wing, n_sw, n_cw):

    # --------------------
    # Process VD strengths
    # --------------------   
    bound_vortices = Data()
    bound_vortices.coordinates = []
    bound_vortices.vortex_strengths = []

    left_trailing_vortices = Data()
    left_trailing_vortices.coordinates = []
    left_trailing_vortices.vortex_strengths =[]  
    
    right_trailing_vortices = Data()
    right_trailing_vortices.coordinates = []
    right_trailing_vortices.vortex_strengths =[]      
    
    inf_trailing_vortices = Data()
    inf_trailing_vortices.coordinates = []
    inf_trailing_vortices.vortex_strengths = []
    
    
    # Loop over each panel
    for s in range(n_sw):
        rtv_strength = 0
        
        for c in range(n_cw):
            i = c + s*(n_cw) # Panel number
            
            if s==n_sw-1:
                # Panel is at the wing tip
                g_c_s = wing.gamma[i]
                rtv_strength += g_c_s
            else:
                g_c_s = wing.gamma[i]
                g_c_sp = wing.gamma[i+n_cw]
                rtv_strength += g_c_s - g_c_sp
                
            if c==n_cw-1:
                # right trailing vortex goes back farther
                rtv = np.array([wing.XB_TE[i], wing.YB_TE[i], wing.ZB_TE[i]])   # right bound vortex of next chordwise panel 
                
                # extend one spanwise distance downstream
                x     = np.abs(wing.YB2[0::n_cw][-1]) # one span distance
                rinf  = np.array([rtv[0]+x, rtv[1], rtv[2]])
                
                ## TO DO: rotate to leave trailing edge at freestream
                #rot_mat = np.array([[np.cos(alpha), 0, np.sin(alpha)], [0,1,0], [-np.sin(alpha), 0, np.cos(alpha)]])
                #rinf = np.matmul(rot_mat,rinf)
                
                # save trailing infinite vortices (shortened after 1 spanwise downstream distance)
                inf_trailing_vortices.coordinates.append(rtv)
                inf_trailing_vortices.coordinates.append(rinf)
                inf_trailing_vortices.vortex_strengths.append(rtv_strength)
                
            else:
                rtv = np.array([wing.XBH[i+1], wing.YBH[i+1], wing.ZBH[i+1]])   # right bound vortex of next chordwise panel 
            
            # coordinates for bound vortex and right trailing vortex (left trailing vortex is 0 at symmetry, equals rtv of (c,s+1) panel elsewhere)
            lbv = np.array([wing.XAH[i], wing.YAH[i], wing.ZAH[i]])         # left bound vortex of current panel
            rbv = np.array([wing.XBH[i], wing.YBH[i], wing.ZBH[i]])         # right bound vortex of current panel
              
            
            bound_vortices.coordinates.append(lbv)
            bound_vortices.coordinates.append(rbv)
            bound_vortices.vortex_strengths.append(g_c_s) # bound vortex strength is the vortex strength associated with current panel
            
            right_trailing_vortices.coordinates.append(rbv)
            right_trailing_vortices.coordinates.append(rtv)                   
            right_trailing_vortices.vortex_strengths.append( rtv_strength ) # right trailing vortex strength is zero (symmetry plane)
    
    return right_trailing_vortices, bound_vortices, inf_trailing_vortices






def wake_VD(filename,n_cp,n_sw,inf_trailing_vortices):
    """
    Print and save VTK file for the vortex filaments in the wake of the lifting surface (keeps
    trailing vortices separate from lifting surface filaments)
    """        
    with open(filename, 'w') as f:
        
        #---------------------
        # Write header
        #---------------------
        header = ["# vtk DataFile Version 4.0" ,     # File version and identifier
                  "\nVortex distribution " ,         # Title
                  "\nASCII"              ,           # Data type
                  "\nDATASET UNSTRUCTURED_GRID"    ] # Dataset structure / topology
        f.writelines(header)   
        
        # --------------------
        # Write Points
        # --------------------   
        n_pts = 2*n_sw  # total number of node vertices 
        points_header = "\n\nPOINTS "+str(n_pts) +" float"
        f.write(points_header)     

        for i in range(n_sw):
            # append trailing vortex lines
            p1 = inf_trailing_vortices.coordinates[2*i]
            p2 = inf_trailing_vortices.coordinates[2*i+1]
            f.write("\n"+str(p1[0])+" "+str(p1[1])+" "+str(p1[2]))
            f.write("\n"+str(p2[0])+" "+str(p2[1])+" "+str(p2[2]))
            
        #---------------------    
        # Write Cells:
        #---------------------     
        n          = n_sw
        v_per_cell = 2
        size       = n*(1+v_per_cell)
        
        cell_header  = "\n\nCELLS "+str(n)+" "+str(size)
        f.write(cell_header)         
           
        for i in range(n_sw):
            # write the cells for the inf vortices for each panel
            base_node = i*2
            new_poly_line = "\n2 "+str(base_node)+" "+str(base_node+1)
            f.write(new_poly_line )
            
        #---------------------        
        # Write Cell Types:
        #---------------------
        cell_type_header  = "\n\nCELL_TYPES "+str(n)
        f.write(cell_type_header)        
        
        for i in range(n_sw):
            # inf vortex cell type
            f.write("\n4")
        #--------------------------        
        # Write VTK Poly Line Data:
        #--------------------------
        cell_data_header  = "\n\nCELL_DATA "+str(n)
        f.write(cell_data_header)      
        
        # First scalar value
        f.write("\nSCALARS i float 1")
        f.write("\nLOOKUP_TABLE default")   

        for i in range(n_sw):
            # inf trailing vortex scalar i
            f.write("\n"+str(i))   
            
        # Second scalar value
        f.write("\nSCALARS circulation float 1")
        f.write("\nLOOKUP_TABLE default")   
        
        for i in range(n_sw):
            # inf trailing vortex circulation
            f.write("\n"+str(inf_trailing_vortices.vortex_strengths[i]))        
            
    f.close()
    return

def wing_VD(filename,n_sw,n_cp,bound_vortices, right_trailing_vortices):
    """
    Print and save VTK file for the vortex filaments on the lifting surface (keeps
    trailing vortices separate)
    """    
    with open(filename, 'w') as f:
        #---------------------
        # Write header
        #---------------------
        header = ["# vtk DataFile Version 4.0" ,     # File version and identifier
                  "\nWing panel vortex distribution " ,         # Title
                  "\nASCII"              ,           # Data type
                  "\nDATASET UNSTRUCTURED_GRID"    ] # Dataset structure / topology
        f.writelines(header)   
        
        # --------------------
        # Write Points
        # --------------------   
        pts_per_panel = 4
        n_pts = n_cp*pts_per_panel  # total number of node vertices 
        points_header = "\n\nPOINTS "+str(n_pts) +" float"
        f.write(points_header)     
        for i in range(n_cp):
            # write the bound vortices for each panel
            p1 = bound_vortices.coordinates[2*i]
            p2 = bound_vortices.coordinates[2*i+1]
            f.write("\n"+str(p1[0])+" "+str(p1[1])+" "+str(p1[2]))
            f.write("\n"+str(p2[0])+" "+str(p2[1])+" "+str(p2[2]))
            
            # write the right trailing edge vortex for each panel
            p1 = right_trailing_vortices.coordinates[2*i]
            p2 = right_trailing_vortices.coordinates[2*i+1]
            f.write("\n"+str(p1[0])+" "+str(p1[1])+" "+str(p1[2]))
            f.write("\n"+str(p2[0])+" "+str(p2[1])+" "+str(p2[2]))
            
        #---------------------    
        # Write Cells:
        #---------------------     
        n          = 2*n_cp
        v_per_cell = 2
        size       = n*(1+v_per_cell)
        
        cell_header  = "\n\nCELLS "+str(n)+" "+str(size)
        f.write(cell_header)
        for i in range(n_cp):
            # write the cells for the bound vortices for each panel
            base_node = i*pts_per_panel
            new_poly_line = "\n2 "+str(base_node)+" "+str(base_node+1)
            f.write(new_poly_line )
            
            # write the cells for the right trailing edge vortex for each panel
            base_node = i*pts_per_panel +2
            new_poly_line = "\n2 "+str(base_node)+" "+str(base_node+1)
            f.write(new_poly_line )            
           
            
        #---------------------        
        # Write Cell Types:
        #---------------------
        cell_type_header  = "\n\nCELL_TYPES "+str(n)
        f.write(cell_type_header)        
        for i in range(n_cp):
            # bound vortex cell type
            f.write("\n4")
            # right trailing vortex cell type
            f.write("\n4")
        
        #--------------------------        
        # Write VTK Poly Line Data:
        #--------------------------
        cell_data_header  = "\n\nCELL_DATA "+str(n)
        f.write(cell_data_header)      
        
        # First scalar value
        f.write("\nSCALARS i float 1")
        f.write("\nLOOKUP_TABLE default")   
        
        for i in range(n_cp):
            # bound vortex scalar i
            f.write("\n"+str(i))   
    
            # right trailing vortex scalar i
            f.write("\n"+str(i))         
            
        # Second scalar value
        f.write("\nSCALARS circulation float 1")
        f.write("\nLOOKUP_TABLE default")   
        
        for i in range(n_cp):
            # bound vortex scalar circulation
            f.write("\n"+str(bound_vortices.vortex_strengths[i])) 
            
            # right trailing vortex circulation
            f.write("\n"+str(right_trailing_vortices.vortex_strengths[i]))        
            
    f.close()
    return    



def full_VD(filename, n_cp, n_sw, bound_vortices, right_trailing_vortices, inf_trailing_vortices):
    """
    Print and save VTK file for the entire vortex distribution (including filaments on wing and 
    trailing vortices)
    """
    with open(filename, 'w') as f:

        #---------------------
        # Write header
        #---------------------
        header = ["# vtk DataFile Version 4.0" ,     # File version and identifier
                  "\nVortex distribution " ,         # Title
                  "\nASCII"              ,           # Data type
                  "\nDATASET UNSTRUCTURED_GRID"    ] # Dataset structure / topology
        f.writelines(header)   
        
        # --------------------
        # Write Points
        # --------------------   
        pts_per_panel = 4
        n_pts = n_cp*pts_per_panel + 2*n_sw  # total number of node vertices 
        points_header = "\n\nPOINTS "+str(n_pts) +" float"
        f.write(points_header)     
        for i in range(n_cp):
            # write the bound vortices for each panel
            p1 = bound_vortices.coordinates[2*i]
            p2 = bound_vortices.coordinates[2*i+1]
            f.write("\n"+str(p1[0])+" "+str(p1[1])+" "+str(p1[2]))
            f.write("\n"+str(p2[0])+" "+str(p2[1])+" "+str(p2[2]))
            
            # write the right trailing edge vortex for each panel
            p1 = right_trailing_vortices.coordinates[2*i]
            p2 = right_trailing_vortices.coordinates[2*i+1]
            f.write("\n"+str(p1[0])+" "+str(p1[1])+" "+str(p1[2]))
            f.write("\n"+str(p2[0])+" "+str(p2[1])+" "+str(p2[2]))
        
        for i in range(n_sw):
            # append trailing vortex lines
            p1 = inf_trailing_vortices.coordinates[2*i]
            p2 = inf_trailing_vortices.coordinates[2*i+1]
            f.write("\n"+str(p1[0])+" "+str(p1[1])+" "+str(p1[2]))
            f.write("\n"+str(p2[0])+" "+str(p2[1])+" "+str(p2[2]))
            
        #---------------------    
        # Write Cells:
        #---------------------     
        n          = 2*n_cp+n_sw
        v_per_cell = 2
        size       = n*(1+v_per_cell)
        
        cell_header  = "\n\nCELLS "+str(n)+" "+str(size)
        f.write(cell_header)
        for i in range(n_cp):
            # write the cells for the bound vortices for each panel
            base_node = i*pts_per_panel
            new_poly_line = "\n2 "+str(base_node)+" "+str(base_node+1)
            f.write(new_poly_line )
            
            # write the cells for the right trailing edge vortex for each panel
            base_node = i*pts_per_panel +2
            new_poly_line = "\n2 "+str(base_node)+" "+str(base_node+1)
            f.write(new_poly_line )            
           
        for i in range(n_sw):
            # write the cells for the inf vortices for each panel
            base_node = 4*n_cp + i*2
            new_poly_line = "\n2 "+str(base_node)+" "+str(base_node+1)
            f.write(new_poly_line )
            
        #---------------------        
        # Write Cell Types:
        #---------------------
        cell_type_header  = "\n\nCELL_TYPES "+str(n)
        f.write(cell_type_header)        
        for i in range(n_cp):
            # bound vortex cell type
            f.write("\n4")
            # right trailing vortex cell type
            f.write("\n4")
        
        for i in range(n_sw):
            # inf vortex cell type
            f.write("\n4")
        #--------------------------        
        # Write VTK Poly Line Data:
        #--------------------------
        cell_data_header  = "\n\nCELL_DATA "+str(n)
        f.write(cell_data_header)      
        
        # First scalar value
        f.write("\nSCALARS i float 1")
        f.write("\nLOOKUP_TABLE default")   
        
        for i in range(n_cp):
            # bound vortex scalar i
            f.write("\n"+str(i))   
    
            # right trailing vortex scalar i
            f.write("\n"+str(i))            
            
        for i in range(n_sw):
            # inf trailing vortex scalar i
            f.write("\n"+str(i))   
            
        # Second scalar value
        f.write("\nSCALARS circulation float 1")
        f.write("\nLOOKUP_TABLE default")   
        
        for i in range(n_cp):
            # bound vortex scalar circulation
            f.write("\n"+str(bound_vortices.vortex_strengths[i])) 
            
            # right trailing vortex circulation
            f.write("\n"+str(right_trailing_vortices.vortex_strengths[i]))  
            
        for i in range(n_sw):
            # inf trailing vortex circulation
            f.write("\n"+str(inf_trailing_vortices.vortex_strengths[i]))        
            
    f.close()
    return