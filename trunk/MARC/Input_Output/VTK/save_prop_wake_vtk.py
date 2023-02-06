## @ingroup Input_Output-VTK
# save_prop_wake_vtk.py
# 
# Created:    Jun 2021, R. Erhard
# Modified:   Jul 2022, R. Erhard
#     
from MARC.Core import Data      
import numpy as np

## @ingroup Input_Output-VTK
def save_prop_wake_vtk(prop,wVD,gamma,filename,Results,start_angle_idx,origin_offset,rot=-1,aircraftReferenceFrame=True):
    """
    Saves a MARC propeller wake as a VTK in legacy format.

    Inputs:
       wVD           Vortex distribution of propeller wake          [Unitless]  
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
    m = 0 # first control point
    
    # Extract wake properties of the ith propeller
    n_time_steps    = len(wVD.XA1[start_angle_idx,m,0,0,:])
    n_blades        = len(wVD.XA1[start_angle_idx,m,:,0,0])
    n_radial_rings  = len(wVD.XA1[start_angle_idx,m,0,:,0])
    
    
    # Create file
    with open(filename, 'w') as f:
    
        #---------------------
        # Write header
        #---------------------
        header = ["# vtk DataFile Version 4.0"              , # File version and identifier
                  "\nMARC Model of PROWIM Propeller Wake " , # Title
                  "\nASCII"                                 , # Data type
                  "\nDATASET UNSTRUCTURED_GRID"             ] # Dataset structure / topology
        f.writelines(header)   
        
        # --------------------
        # Write Points
        # --------------------   
        n_vertices = n_blades*(n_radial_rings+1)*(n_time_steps+1)    # total number of node vertices
        points_header = "\n\nPOINTS "+str(n_vertices) +" float"
        f.write(points_header)    
        node_number=[] 

        matA1 = np.zeros(np.append(np.shape(wVD.XA1),3))
        matA2 = np.zeros(np.append(np.shape(wVD.XA1),3))        
        matB1 = np.zeros(np.append(np.shape(wVD.XA1),3))
        matB2 = np.zeros(np.append(np.shape(wVD.XA1),3))
        
        matA1[:,:,:,:,:,0] = wVD.XA1 + origin_offset[0]
        matA1[:,:,:,:,:,1] = wVD.YA1 + origin_offset[1]
        matA1[:,:,:,:,:,2] = wVD.ZA1 + origin_offset[2]

        matA2[:,:,:,:,:,0] = wVD.XA2 + origin_offset[0]
        matA2[:,:,:,:,:,1] = wVD.YA2 + origin_offset[1]
        matA2[:,:,:,:,:,2] = wVD.ZA2 + origin_offset[2]

        matB1[:,:,:,:,:,0] = wVD.XB1 + origin_offset[0]
        matB1[:,:,:,:,:,1] = wVD.YB1 + origin_offset[1]
        matB1[:,:,:,:,:,2] = wVD.ZB1 + origin_offset[2]

        matB2[:,:,:,:,:,0] = wVD.XB2 + origin_offset[0]
        matB2[:,:,:,:,:,1] = wVD.YB2 + origin_offset[1]
        matB2[:,:,:,:,:,2] = wVD.ZB2 + origin_offset[2]
        
        if aircraftReferenceFrame:
            # rotate points to aircraft frame
            trans_3 =  prop.prop_vel_to_body() 
            matA1 = np.matmul(matA1, trans_3)
            matA2 = np.matmul(matA2, trans_3)
            matB1 = np.matmul(matB1, trans_3)
            matB2 = np.matmul(matB2, trans_3)
        
        
        if rot ==1:
            # Flip around to use A's as right-side of panel, 1=LE, 2=TE
            XA2 = matB2[:,:,:,:,:,0]  # bottom left corner of panel
            XB2 = matA2[:,:,:,:,:,0]  # bottom right corner of panel
            XA1 = matB1[:,:,:,:,:,0]  # top left corner of panel
            XB1 = matA1[:,:,:,:,:,0]  # top right corner of panel    
            YA2 = matB2[:,:,:,:,:,1]  # bottom left corner of panel
            YB2 = matA2[:,:,:,:,:,1]  # bottom right corner of panel
            YA1 = matB1[:,:,:,:,:,1]  # top left corner of panel
            YB1 = matA1[:,:,:,:,:,1]  # top right corner of panel   
            ZA2 = matB2[:,:,:,:,:,2]  # bottom left corner of panel
            ZB2 = matA2[:,:,:,:,:,2]  # bottom right corner of panel
            ZA1 = matB1[:,:,:,:,:,2]  # top left corner of panel
            ZB1 = matA1[:,:,:,:,:,2]  # top right corner of panel               
        else:
            # Use B's as rightmost panel
            XA2 = matA2[:,:,:,:,:,0]  # bottom left corner of panel
            XB2 = matB2[:,:,:,:,:,0]  # bottom right corner of panel
            XA1 = matA1[:,:,:,:,:,0]  # top left corner of panel
            XB1 = matB1[:,:,:,:,:,0]  # top right corner of panel    
            YA2 = matA2[:,:,:,:,:,1]  # bottom left corner of panel
            YB2 = matB2[:,:,:,:,:,1]  # bottom right corner of panel
            YA1 = matA1[:,:,:,:,:,1]  # top left corner of panel
            YB1 = matB1[:,:,:,:,:,1]  # top right corner of panel   
            ZA2 = matA2[:,:,:,:,:,2]  # bottom left corner of panel
            ZB2 = matB2[:,:,:,:,:,2]  # bottom right corner of panel
            ZA1 = matA1[:,:,:,:,:,2]  # top left corner of panel
            ZB1 = matB1[:,:,:,:,:,2]  # top right corner of panel              
           
        
        # Loop over number of rotor blades
        for B_idx in range(n_blades):
            # Loop over number of "chordwise" panels in the wake distribution (t0 is most recently shed from blade)
            for t_idx in range(n_time_steps+1):
                # Loop over number of "radial" or "spanwise" panels in the wake distribution 
                for r_idx in range(n_radial_rings+1):            
                    
                    #-------------------------------------------------------------------
                    # Get vertices for each node
                    #-------------------------------------------------------------------
                    if r_idx == n_radial_rings and t_idx==0:
                        # Last ring at t0; use B2 to get rightmost TE node
                        x = round(XB2[start_angle_idx,m,B_idx,r_idx-1,t_idx],4)
                        y = round(YB2[start_angle_idx,m,B_idx,r_idx-1,t_idx],4)
                        z = round(ZB2[start_angle_idx,m,B_idx,r_idx-1,t_idx],4)
                        
                    elif t_idx==0:
                        # First set of rings shed; use A2 to get left TE node
                        x = round(XA2[start_angle_idx,m,B_idx,r_idx,t_idx],4)
                        y = round(YA2[start_angle_idx,m,B_idx,r_idx,t_idx],4)
                        z = round(ZA2[start_angle_idx,m,B_idx,r_idx,t_idx],4)   
                        
                    elif r_idx==n_radial_rings:  
                        # Last radial ring for tstep; use B1 of prior to get tip node
                        x = round(XB1[start_angle_idx,m,B_idx,r_idx-1,t_idx-1],4)
                        y = round(YB1[start_angle_idx,m,B_idx,r_idx-1,t_idx-1],4)
                        z = round(ZB1[start_angle_idx,m,B_idx,r_idx-1,t_idx-1],4)
                    else:
                        # print the point index (Left LE --> Left TE --> Right LE --> Right TE)
                        x = round(XA1[start_angle_idx,m,B_idx,r_idx,t_idx-1],4)
                        y = round(YA1[start_angle_idx,m,B_idx,r_idx,t_idx-1],4)
                        z = round(ZA1[start_angle_idx,m,B_idx,r_idx,t_idx-1],4)
                    
                    new_point = "\n"+str(x)+" "+str(y)+" "+str(z)
                    node_number = np.append(node_number, r_idx + (n_radial_rings+1)*t_idx)
                    #print(new_point)
                    f.write(new_point)                
        
            
        #---------------------    
        # Write Cells:
        #---------------------
        cells_per_blade = n_radial_rings*n_time_steps
        n_cells         = n_blades*cells_per_blade # total number of cells
        v_per_cell      = 4 # quad cells
        size            = n_cells*(1+v_per_cell) # total number of integer values required to represent the list
        cell_header     = "\n\nCELLS "+str(n_cells)+" "+str(size)
        f.write(cell_header)
        
        for B_idx in range(n_blades):
            for i in range(cells_per_blade):
                if i==0:
                    node =  i + int(B_idx*n_vertices/n_blades)
                elif i%n_radial_rings ==0:
                    node = node+1
                new_cell = "\n4 "+str(node)+" "+str(node+1)+" "+str(node+n_radial_rings+2)+" "+str(node+n_radial_rings+1)
                f.write(new_cell)
                #print(new_cell)
                # update node:
                node = node+1 
        
        #---------------------        
        # Write Cell Types:
        #---------------------
        cell_type_header  = "\n\nCELL_TYPES "+str(n_cells)
        f.write(cell_type_header)        
        for i in range(n_cells):
            f.write("\n9")      
            
        #--------------------------        
        # Write Scalar Cell Data:
        #--------------------------
        cell_data_header  = "\n\nCELL_DATA "+str(n_cells)
        f.write(cell_data_header)      
        
        # Zero-th scalar value
        f.write("\nSCALARS i float 1")
        f.write("\nLOOKUP_TABLE default")   

        for i in range(n_cells):
            f.write("\n"+str(i))             
        
        # First scalar value
        f.write("\nSCALARS gamma float 1")
        f.write("\nLOOKUP_TABLE default")   
        
        circulations=[]
        for B_idx in range(n_blades):
            for t_idx in range(n_time_steps):
                for r_idx in range(n_radial_rings):
                    circulations = np.append(circulations, gamma[0,B_idx,r_idx,t_idx])     
                    
        for i in range(n_cells):
            new_circ = str(circulations[i])
            f.write("\n"+new_circ)     

        
        # Second scalar value
        f.write("\nSCALARS vt float 1")
        f.write("\nLOOKUP_TABLE default")   
        vt = Results['prop_outputs'].blade_tangential_induced_velocity[0]
        for B_idx in range(n_blades):
            for i in range(cells_per_blade):
                vt_L = vt[int(i%(n_radial_rings))]
                vt_R = vt[int(i%(n_radial_rings))+1]
                vt_C = 0.5*(vt_L+vt_R)
                
                new_vt = str(vt_C)
                f.write("\n"+new_vt)                  
    f.close()

    # Loop over number of rotor blades
    for B_idx in range(n_blades):
        rings = Data()
        rings.coordinates = []
        rings.vortex_strengths = []
        
        # Loop over number of "chordwise" panels in the wake distribution
        for t_idx in range(n_time_steps):
            g        = gamma[m,B_idx,:,t_idx] # circulation distribution on current blade at current timestep
            dgamma   = np.gradient(g)            # gradient of the blade circulation distribution
            gamma_slope_sign = np.ones_like(dgamma)
            gamma_slope_sign[dgamma<0] = -1
   
            
            # Loop over number of "radial" or "spanwise" panels in the wake distribution
            for r_idx in range(n_radial_rings):
                
                # Get vortex strength of panel (current node is the bottom left of the panel)
                g_r_t = gamma[m,B_idx,r_idx,t_idx]
                
                p_r_t   = np.array([XA1[start_angle_idx,m,B_idx,r_idx,t_idx],YA1[start_angle_idx,m,B_idx,r_idx,t_idx],ZA1[start_angle_idx,m,B_idx,r_idx,t_idx]])  # Bottom Left
                p_rp_t  = np.array([XB1[start_angle_idx,m,B_idx,r_idx,t_idx],YB1[start_angle_idx,m,B_idx,r_idx,t_idx],ZB1[start_angle_idx,m,B_idx,r_idx,t_idx]])  # Top Left
                p_r_tp  = np.array([XA2[start_angle_idx,m,B_idx,r_idx,t_idx],YA2[start_angle_idx,m,B_idx,r_idx,t_idx],ZA2[start_angle_idx,m,B_idx,r_idx,t_idx]])  # Top Right
                p_rp_tp = np.array([XB2[start_angle_idx,m,B_idx,r_idx,t_idx],YB2[start_angle_idx,m,B_idx,r_idx,t_idx],ZB2[start_angle_idx,m,B_idx,r_idx,t_idx]])  # Bottom Right
                
                # Append vortex strengths to ring vortices
                if t_idx==0 and r_idx==0:
                    # 
                    g_rp_t = gamma[m,B_idx,r_idx+1,t_idx] 
                    g_r_tp = gamma[m,B_idx,r_idx,t_idx+1]   
                    
                    # Bottom edge
                    rings.coordinates.append(p_r_t)        # bottom left node   (Bottom edge)
                    rings.coordinates.append(p_rp_t)       # bottom right node  (Bottom edge)
                    rings.vortex_strengths.append(g_r_t )  # bottom segment of initial ring (only sees the current ring vortex strength)
            
                    # Top edge
                    rings.coordinates.append(p_r_tp)       # top left node   (Bottom edge)
                    rings.coordinates.append(p_rp_tp)      # top right node  (Bottom edge)
                    rings.vortex_strengths.append(g_r_t - g_r_tp) # difference between prior time step ring strength
                                            
                    # Left edge
                    rings.coordinates.append(p_r_t)         # bottom left node  (Left edge)
                    rings.coordinates.append(p_r_tp)        # top left node     (Left edge)
                    rings.vortex_strengths.append(g_r_t)    # left edge of first ring (only current ring vortex)
                    
                    # Right edge
                    rings.coordinates.append(p_rp_t)               # bottom right node  (Right edge)
                    rings.coordinates.append(p_rp_tp)              # top right node     (Right edge)
                    rings.vortex_strengths.append(-gamma_slope_sign[r_idx]*(g_r_t - g_rp_t))  # right segment of ring
                
                
                    
                elif t_idx==0 and r_idx==n_radial_rings-1:
                    
                    g_r_tp = gamma[m,B_idx,r_idx,t_idx+1]
                    
                    # Bottom edge
                    rings.coordinates.append(p_r_t)
                    rings.coordinates.append(p_rp_t)
                    rings.vortex_strengths.append(g_r_t ) # bottom segment of initial ring (only sees the current ring vortex strength)
                    
                    # Top edge
                    rings.coordinates.append(p_r_tp)       # top left node   (Bottom edge)
                    rings.coordinates.append(p_rp_tp)      # tp right node  (Bottom edge)
                    rings.vortex_strengths.append(g_r_t - g_r_tp) # bottom segment of initial ring (only sees the current ring vortex strength)
                    
                    # Right edge
                    rings.coordinates.append(p_rp_t)
                    rings.coordinates.append(p_rp_tp)   
                    rings.vortex_strengths.append(g_r_t )  # right segment of tip ring (only has current ring vortex)                    
                
                elif t_idx==0:
                    #     
                    g_rp_t = gamma[m,B_idx,r_idx+1,t_idx] 
                    g_r_tp = gamma[m,B_idx,r_idx,t_idx+1]
                    
                    # Bottom edge
                    rings.coordinates.append(p_r_t)       # bottom left node   (Bottom edge)
                    rings.coordinates.append(p_rp_t)      # bottom right node  (Bottom edge)
                    rings.vortex_strengths.append(g_r_t ) # bottom segment of initial ring (only sees the current ring vortex strength)
            
                    # Top edge
                    rings.coordinates.append(p_r_tp)       # top left node   (Bottom edge)
                    rings.coordinates.append(p_rp_tp)      # tp right node  (Bottom edge)
                    rings.vortex_strengths.append(g_r_t - g_r_tp) # bottom segment of initial ring (only sees the current ring vortex strength)
                                            
                    # Right edge
                    rings.coordinates.append(p_rp_t)               # bottom right node  (Right edge)
                    rings.coordinates.append(p_rp_tp)              # top right node     (Right edge)
                    rings.vortex_strengths.append(-gamma_slope_sign[r_idx]*(g_r_t - g_rp_t))  # right segment of ring                        
                
                elif t_idx==(n_time_steps-1) and r_idx==0:
                    #  
                    g_rp_t = gamma[m,B_idx,r_idx+1,t_idx]                        
                    # Top edge
                    rings.coordinates.append(p_r_tp)       # top left node   (Bottom edge)
                    rings.coordinates.append(p_rp_tp)      # top right node  (Bottom edge)
                    rings.vortex_strengths.append(g_r_t )  # top segment of ring 
                    
                    # Left edge
                    rings.coordinates.append(p_r_t)      # bottom left node  (Left edge)
                    rings.coordinates.append(p_r_tp)     # top left node     (Left edge)
                    rings.vortex_strengths.append(g_r_t) # left edge of first ring (only current ring vortex)
                    
                    # Right edge
                    rings.coordinates.append(p_rp_t)               # bottom right node  (Right edge)
                    rings.coordinates.append(p_rp_tp)              # top right node     (Right edge)
                    rings.vortex_strengths.append(-gamma_slope_sign[r_idx]*(g_r_t - g_rp_t))  # right segment of ring                
                elif r_idx==0:
                    # 
                    g_rp_t = gamma[m,B_idx,r_idx+1,t_idx]    
                    g_r_tp = gamma[m,B_idx,r_idx,t_idx+1]                        
                    # Top edge
                    rings.coordinates.append(p_r_tp)       # top left node   (Bottom edge)
                    rings.coordinates.append(p_rp_tp)      # top right node  (Bottom edge)
                    rings.vortex_strengths.append(g_r_t - g_r_tp ) # top segment of ring 
                    
                    # Left edge
                    rings.coordinates.append(p_r_t)      # bottom left node  (Left edge)
                    rings.coordinates.append(p_r_tp)     # top left node     (Left edge)
                    rings.vortex_strengths.append(g_r_t) # left edge of first ring (only current ring vortex)
                    
                    # Right edge
                    rings.coordinates.append(p_rp_t)               # bottom right node  (Right edge)
                    rings.coordinates.append(p_rp_tp)              # top right node     (Right edge)
                    rings.vortex_strengths.append(-gamma_slope_sign[r_idx]*(g_r_t - g_rp_t))  # right segment of ring
                
                elif t_idx==(n_time_steps-1) and r_idx==(n_radial_rings-1):
                    # 
                    # Top edge
                    rings.coordinates.append(p_r_tp)       # top left node   (Bottom edge)
                    rings.coordinates.append(p_rp_tp)      # top right node  (Bottom edge)
                    rings.vortex_strengths.append(g_r_t )  # top segment of ring 
                    
                    # Right edge
                    rings.coordinates.append(p_rp_t)               # bottom right node  (Right edge)
                    rings.coordinates.append(p_rp_tp)              # top right node     (Right edge)
                    rings.vortex_strengths.append(g_r_t )  # right segment of ring     
                    
                elif r_idx==n_radial_rings-1:
                     
                    g_r_tp = gamma[m,B_idx,r_idx,t_idx+1]                        
                    # Top edge
                    rings.coordinates.append(p_r_tp)       # top left node   (Bottom edge)
                    rings.coordinates.append(p_rp_tp)      # top right node  (Bottom edge)
                    rings.vortex_strengths.append(g_r_t - g_r_tp ) # top segment of ring 
            
                    # Right edge
                    rings.coordinates.append(p_rp_t)      # bottom right node  (Right edge)
                    rings.coordinates.append(p_rp_tp)     # top right node     (Right edge)
                    rings.vortex_strengths.append(g_r_t)  # right segment of ring         

                elif t_idx==(n_time_steps-1):
                    # 
                    g_rp_t = gamma[m,B_idx,r_idx+1,t_idx]
                    # Top edge
                    rings.coordinates.append(p_r_tp)       # top left node   (Bottom edge)
                    rings.coordinates.append(p_rp_tp)      # top right node  (Bottom edge)
                    rings.vortex_strengths.append(g_r_t  )  # top segment of ring 
                    
                    # Right edge
                    rings.coordinates.append(p_rp_t)               # bottom right node  (Right edge)
                    rings.coordinates.append(p_rp_tp)              # top right node     (Right edge)
                    rings.vortex_strengths.append(-gamma_slope_sign[r_idx]*(g_r_t - g_rp_t))  # right segment of ring                            
                           
                else:           
                    g_rp_t = gamma[m,B_idx,r_idx+1,t_idx]
                    g_r_tp = gamma[m,B_idx,r_idx,t_idx+1]    
                    
                    # Top edge
                    rings.coordinates.append(p_r_tp)               # top left node   (Bottom edge)
                    rings.coordinates.append(p_rp_tp)              # top right node  (Bottom edge)
                    rings.vortex_strengths.append(g_r_t - g_r_tp ) # top segment of ring 
            
                    # Right edge
                    rings.coordinates.append(p_rp_t)               # bottom right node  (Right edge)
                    rings.coordinates.append(p_rp_tp)              # top right node     (Right edge)
                    rings.vortex_strengths.append(-gamma_slope_sign[r_idx]*(g_r_t - g_rp_t))  # right segment of ring    
                    
                
        # Store vortex distribution for this blade
        sep  = filename.rfind('_')
        VD_filename = filename[0:sep]+"_VD_blade"+str(B_idx)+filename[sep:]  
        write_VD(rings,n_time_steps,n_radial_rings, VD_filename)
    return

def write_VD(rings, nt,nr, filename):
    
    
    # Create file
    with open(filename, 'w') as f:    
        #---------------------
        # Write header
        #---------------------
        header = ["# vtk DataFile Version 4.0" ,     # File version and identifier
                  "\nWake vortex distribution " ,    # Title
                  "\nASCII"              ,           # Data type
                  "\nDATASET UNSTRUCTURED_GRID"    ] # Dataset structure / topology
        f.writelines(header)   
        
        # --------------------
        # Write Points
        # --------------------   
        n_edges       = nr*(nt+1) + (nt*(nr+1))
        pts_per_edge  = 2
        n_pts         = n_edges*pts_per_edge # total number of node vertices 
        points_header = "\n\nPOINTS "+str(n_pts) +" float"
        
        f.write(points_header)     
        for i in range(n_edges):
            # write the side vortices for each panel
            p1 = rings.coordinates[2*i]
            p2 = rings.coordinates[2*i+1]
            f.write("\n"+str(p1[0])+" "+str(p1[1])+" "+str(p1[2]))
            f.write("\n"+str(p2[0])+" "+str(p2[1])+" "+str(p2[2]))
        
        
        #---------------------    
        # Write Cells:
        #---------------------    
        v_per_cell = 2
        size       = n_edges*(1+v_per_cell)
        
        cell_header  = "\n\nCELLS "+str(n_edges)+" "+str(size)
        f.write(cell_header)
        for i in range(n_edges):
            # write the cells for the side vortices for each panel
            base_node = i*2
            new_poly_line = "\n2 "+str(base_node)+" "+str(base_node+1)
            f.write(new_poly_line )    
    
        #---------------------        
        # Write Cell Types:
        #---------------------
        cell_type_header  = "\n\nCELL_TYPES "+str(n_edges)
        f.write(cell_type_header)        
        for i in range(n_edges):
            # cell type
            f.write("\n4")
            
        #--------------------------        
        # Write VTK Poly Line Data:
        #--------------------------
        cell_data_header  = "\n\nCELL_DATA "+str(n_edges)
        f.write(cell_data_header)      
        
        # First scalar value
        f.write("\nSCALARS i float 1")
        f.write("\nLOOKUP_TABLE default")   
        
        for i in range(n_edges):
            # vortex scalar i
            f.write("\n"+str(i))   
        # Second scalar value
        f.write("\nSCALARS circulation float 1")
        f.write("\nLOOKUP_TABLE default")   
    
        ring_circulations = rings.vortex_strengths
        
        # flag to zero-out lifting line panel (for visualizing shed vortices)
        zero_llps = True  
        if zero_llps:
            ring_circulations = np.array(ring_circulations)
            
            bools = np.zeros_like(ring_circulations).astype(bool)
            bools[0] = True
            bools[4::3][0:nr-1] = True
            
            
            ring_circulations[bools] = 0
            
        for i in range(n_edges):
            # bound vortex scalar circulation
            f.write("\n"+str(ring_circulations[i])) 
                    
    return