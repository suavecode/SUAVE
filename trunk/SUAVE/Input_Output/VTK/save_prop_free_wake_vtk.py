## @ingroup Input_Output-VTK
# save_prop_wake_vtk.py
# 
# Created:    Jun 2021, R. Erhard
# Modified: 
#     
from SUAVE.Core import Data      
import numpy as np

def save_prop_free_wake_vtk(VD,gamma,filename,Results,i_prop):
    """
    Saves a SUAVE propeller wake as a VTK in legacy format.

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
    wVD = VD.Wake
    # Extract wake properties of the ith propeller
    n_time_steps    = len(wVD.XA1[i_prop,0,0,:])
    n_blades        = len(wVD.XA1[i_prop,:,0,0])
    n_radial_rings  = len(wVD.XA1[i_prop,0,:,0])
    
    # Create file
    with open(filename, 'w') as f:
    
        #---------------------
        # Write header
        #---------------------
        header = ["# vtk DataFile Version 4.0"              , # File version and identifier
                  "\nSUAVE Model of PROWIM Propeller Wake " , # Title
                  "\nASCII"                                 , # Data type
                  "\nDATASET UNSTRUCTURED_GRID"             ] # Dataset structure / topology
        f.writelines(header)   
        
        # --------------------
        # Write Points
        # --------------------   
        n_vertices = np.size(wVD.XC)*4    # total number of node vertices
        points_header = "\n\nPOINTS "+str(n_vertices) +" float"
        f.write(points_header)    

        cells  = []
        gammas = []
        i=0
        for p in range(len(VD.Wake_collapsed.XC[0])):
            xA1 = VD.Wake_collapsed.XA1[0][p]
            yA1 = VD.Wake_collapsed.YA1[0][p]
            zA1 = VD.Wake_collapsed.ZA1[0][p]
            
            xA2 = VD.Wake_collapsed.XA2[0][p]
            yA2 = VD.Wake_collapsed.YA2[0][p]
            zA2 = VD.Wake_collapsed.ZA2[0][p]   
            
            xB1 = VD.Wake_collapsed.XB1[0][p]
            yB1 = VD.Wake_collapsed.YB1[0][p]
            zB1 = VD.Wake_collapsed.ZB1[0][p]   
            
            xB2 = VD.Wake_collapsed.XB2[0][p]
            yB2 = VD.Wake_collapsed.YB2[0][p]
            zB2 = VD.Wake_collapsed.ZB2[0][p]  
            
            # print nodes A1 --> A2 --> B2 --> B1
            f.write("\n"+str(xA1)+" "+str(yA1)+" "+str(zA1))    
            f.write("\n"+str(xA2)+" "+str(yA2)+" "+str(zA2))  
            f.write("\n"+str(xB2)+" "+str(yB2)+" "+str(zB2))  
            f.write("\n"+str(xB1)+" "+str(yB1)+" "+str(zB1))            
        
        
            cells.append("\n4 "+str(i)+ " "+str(i+1)+ " "+str(i+2)+" "+ str(i+3))
            gammas.append(gamma[0][p])
            
            i+=4
            
        #---------------------    
        # Write Cells:
        #---------------------
        n_cells         = len(cells) # total number of cells
        v_per_cell      = 4 # quad cells
        size            = n_cells*(1+v_per_cell) # total number of integer values required to represent the list
        cell_header     = "\n\nCELLS "+str(n_cells)+" "+str(size)
        f.write(cell_header)
        
        for i in range(len(cells)):
            new_cell = cells[i]
            f.write(new_cell)

        
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
        
        # First scalar value
        f.write("\nSCALARS gamma float 1")
        f.write("\nLOOKUP_TABLE default")   
        for i in range(n_cells):
            new_circ = str(gammas[i])
            f.write("\n"+new_circ)     
        
        # Second scalar value
        f.write("\nSCALARS i float 1")
        f.write("\nLOOKUP_TABLE default")   
        for i in range(n_cells):
            f.write("\n"+str(i))
            
    f.close()

    ## Loop over number of rotor blades
    #for B_idx in range(n_blades):
        #rings = Data()
        #rings.coordinates = []
        #rings.vortex_strengths = []
        
        ## Loop over number of "chordwise" panels in the wake distribution
        #for t_idx in range(n_time_steps):
            #g        = gamma[i_prop,B_idx,:,t_idx] # circulation distribution on current blade at current timestep
            #dgamma   = np.gradient(g)            # gradient of the blade circulation distribution
            #gamma_slope_sign = np.ones_like(dgamma)
            #gamma_slope_sign[dgamma<0] = -1
   
            
            ## Loop over number of "radial" or "spanwise" panels in the wake distribution
            #for r_idx in range(n_radial_rings):
                
                ## Get vortex strength of panel (current node is the bottom left of the panel)
                #g_r_t = gamma[i_prop,B_idx,r_idx,t_idx]
                
                #p_r_t   = np.array([wVD.XA1[i_prop,B_idx,r_idx,t_idx],wVD.YA1[i_prop,B_idx,r_idx,t_idx],wVD.ZA1[i_prop,B_idx,r_idx,t_idx]])  # Bottom Left
                #p_rp_t  = np.array([wVD.XB1[i_prop,B_idx,r_idx,t_idx],wVD.YB1[i_prop,B_idx,r_idx,t_idx],wVD.ZB1[i_prop,B_idx,r_idx,t_idx]])  # Top Left
                #p_r_tp  = np.array([wVD.XA2[i_prop,B_idx,r_idx,t_idx],wVD.YA2[i_prop,B_idx,r_idx,t_idx],wVD.ZA2[i_prop,B_idx,r_idx,t_idx]])  # Top Right
                #p_rp_tp = np.array([wVD.XB2[i_prop,B_idx,r_idx,t_idx],wVD.YB2[i_prop,B_idx,r_idx,t_idx],wVD.ZB2[i_prop,B_idx,r_idx,t_idx]])  # Bottom Right
                
                ## Append vortex strengths to ring vortices
                #if t_idx==0 and r_idx==0:
                    ## 
                    #g_rp_t = gamma[i_prop,B_idx,r_idx+1,t_idx] 
                    #g_r_tp = gamma[i_prop,B_idx,r_idx,t_idx+1]   
                    
                    ## Bottom edge
                    #rings.coordinates.append(p_r_t)        # bottom left node   (Bottom edge)
                    #rings.coordinates.append(p_rp_t)       # bottom right node  (Bottom edge)
                    #rings.vortex_strengths.append(g_r_t )  # bottom segment of initial ring (only sees the current ring vortex strength)
            
                    ## Top edge
                    #rings.coordinates.append(p_r_tp)       # top left node   (Bottom edge)
                    #rings.coordinates.append(p_rp_tp)      # top right node  (Bottom edge)
                    #rings.vortex_strengths.append(g_r_t - g_r_tp) # difference between prior time step ring strength
                                            
                    ## Left edge
                    #rings.coordinates.append(p_r_t)         # bottom left node  (Left edge)
                    #rings.coordinates.append(p_r_tp)        # top left node     (Left edge)
                    #rings.vortex_strengths.append(g_r_t)    # left edge of first ring (only current ring vortex)
                    
                    ## Right edge
                    #rings.coordinates.append(p_rp_t)               # bottom right node  (Right edge)
                    #rings.coordinates.append(p_rp_tp)              # top right node     (Right edge)
                    #rings.vortex_strengths.append(-gamma_slope_sign[r_idx]*(g_r_t - g_rp_t))  # right segment of ring
                
                
                    
                #elif t_idx==0 and r_idx==n_radial_rings-1:
                    
                    #g_r_tp = gamma[i_prop,B_idx,r_idx,t_idx+1]
                    
                    ## Bottom edge
                    #rings.coordinates.append(p_r_t)
                    #rings.coordinates.append(p_rp_t)
                    #rings.vortex_strengths.append(g_r_t ) # bottom segment of initial ring (only sees the current ring vortex strength)
                    
                    ## Top edge
                    #rings.coordinates.append(p_r_tp)       # top left node   (Bottom edge)
                    #rings.coordinates.append(p_rp_tp)      # tp right node  (Bottom edge)
                    #rings.vortex_strengths.append(g_r_t - g_r_tp) # bottom segment of initial ring (only sees the current ring vortex strength)
                    
                    ## Right edge
                    #rings.coordinates.append(p_rp_t)
                    #rings.coordinates.append(p_rp_tp)   
                    #rings.vortex_strengths.append(g_r_t )  # right segment of tip ring (only has current ring vortex)                    
                
                #elif t_idx==0:
                    ##     
                    #g_rp_t = gamma[i_prop,B_idx,r_idx+1,t_idx] 
                    #g_r_tp = gamma[i_prop,B_idx,r_idx,t_idx+1]
                    
                    ## Bottom edge
                    #rings.coordinates.append(p_r_t)       # bottom left node   (Bottom edge)
                    #rings.coordinates.append(p_rp_t)      # bottom right node  (Bottom edge)
                    #rings.vortex_strengths.append(g_r_t ) # bottom segment of initial ring (only sees the current ring vortex strength)
            
                    ## Top edge
                    #rings.coordinates.append(p_r_tp)       # top left node   (Bottom edge)
                    #rings.coordinates.append(p_rp_tp)      # tp right node  (Bottom edge)
                    #rings.vortex_strengths.append(g_r_t - g_r_tp) # bottom segment of initial ring (only sees the current ring vortex strength)
                                            
                    ## Right edge
                    #rings.coordinates.append(p_rp_t)               # bottom right node  (Right edge)
                    #rings.coordinates.append(p_rp_tp)              # top right node     (Right edge)
                    #rings.vortex_strengths.append(-gamma_slope_sign[r_idx]*(g_r_t - g_rp_t))  # right segment of ring                        
                
                #elif t_idx==(n_time_steps-1) and r_idx==0:
                    ##  
                    #g_rp_t = gamma[i_prop,B_idx,r_idx+1,t_idx]                        
                    ## Top edge
                    #rings.coordinates.append(p_r_tp)       # top left node   (Bottom edge)
                    #rings.coordinates.append(p_rp_tp)      # top right node  (Bottom edge)
                    #rings.vortex_strengths.append(g_r_t )  # top segment of ring 
                    
                    ## Left edge
                    #rings.coordinates.append(p_r_t)      # bottom left node  (Left edge)
                    #rings.coordinates.append(p_r_tp)     # top left node     (Left edge)
                    #rings.vortex_strengths.append(g_r_t) # left edge of first ring (only current ring vortex)
                    
                    ## Right edge
                    #rings.coordinates.append(p_rp_t)               # bottom right node  (Right edge)
                    #rings.coordinates.append(p_rp_tp)              # top right node     (Right edge)
                    #rings.vortex_strengths.append(-gamma_slope_sign[r_idx]*(g_r_t - g_rp_t))  # right segment of ring                
                #elif r_idx==0:
                    ## 
                    #g_rp_t = gamma[i_prop,B_idx,r_idx+1,t_idx]    
                    #g_r_tp = gamma[i_prop,B_idx,r_idx,t_idx+1]                        
                    ## Top edge
                    #rings.coordinates.append(p_r_tp)       # top left node   (Bottom edge)
                    #rings.coordinates.append(p_rp_tp)      # top right node  (Bottom edge)
                    #rings.vortex_strengths.append(g_r_t - g_r_tp ) # top segment of ring 
                    
                    ## Left edge
                    #rings.coordinates.append(p_r_t)      # bottom left node  (Left edge)
                    #rings.coordinates.append(p_r_tp)     # top left node     (Left edge)
                    #rings.vortex_strengths.append(g_r_t) # left edge of first ring (only current ring vortex)
                    
                    ## Right edge
                    #rings.coordinates.append(p_rp_t)               # bottom right node  (Right edge)
                    #rings.coordinates.append(p_rp_tp)              # top right node     (Right edge)
                    #rings.vortex_strengths.append(-gamma_slope_sign[r_idx]*(g_r_t - g_rp_t))  # right segment of ring
                
                #elif t_idx==(n_time_steps-1) and r_idx==(n_radial_rings-1):
                    ## 
                    ## Top edge
                    #rings.coordinates.append(p_r_tp)       # top left node   (Bottom edge)
                    #rings.coordinates.append(p_rp_tp)      # top right node  (Bottom edge)
                    #rings.vortex_strengths.append(g_r_t )  # top segment of ring 
                    
                    ## Right edge
                    #rings.coordinates.append(p_rp_t)               # bottom right node  (Right edge)
                    #rings.coordinates.append(p_rp_tp)              # top right node     (Right edge)
                    #rings.vortex_strengths.append(g_r_t )  # right segment of ring     
                    
                #elif r_idx==n_radial_rings-1:
                     
                    #g_r_tp = gamma[i_prop,B_idx,r_idx,t_idx+1]                        
                    ## Top edge
                    #rings.coordinates.append(p_r_tp)       # top left node   (Bottom edge)
                    #rings.coordinates.append(p_rp_tp)      # top right node  (Bottom edge)
                    #rings.vortex_strengths.append(g_r_t - g_r_tp ) # top segment of ring 
            
                    ## Right edge
                    #rings.coordinates.append(p_rp_t)      # bottom right node  (Right edge)
                    #rings.coordinates.append(p_rp_tp)     # top right node     (Right edge)
                    #rings.vortex_strengths.append(g_r_t)  # right segment of ring         

                #elif t_idx==(n_time_steps-1):
                    ## 
                    #g_rp_t = gamma[i_prop,B_idx,r_idx+1,t_idx]
                    ## Top edge
                    #rings.coordinates.append(p_r_tp)       # top left node   (Bottom edge)
                    #rings.coordinates.append(p_rp_tp)      # top right node  (Bottom edge)
                    #rings.vortex_strengths.append(g_r_t  )  # top segment of ring 
                    
                    ## Right edge
                    #rings.coordinates.append(p_rp_t)               # bottom right node  (Right edge)
                    #rings.coordinates.append(p_rp_tp)              # top right node     (Right edge)
                    #rings.vortex_strengths.append(-gamma_slope_sign[r_idx]*(g_r_t - g_rp_t))  # right segment of ring                            
                           
                #else:           
                    #g_rp_t = gamma[i_prop,B_idx,r_idx+1,t_idx]
                    #g_r_tp = gamma[i_prop,B_idx,r_idx,t_idx+1]    
                    
                    ## Top edge
                    #rings.coordinates.append(p_r_tp)               # top left node   (Bottom edge)
                    #rings.coordinates.append(p_rp_tp)              # top right node  (Bottom edge)
                    #rings.vortex_strengths.append(g_r_t - g_r_tp ) # top segment of ring 
            
                    ## Right edge
                    #rings.coordinates.append(p_rp_t)               # bottom right node  (Right edge)
                    #rings.coordinates.append(p_rp_tp)              # top right node     (Right edge)
                    #rings.vortex_strengths.append(-gamma_slope_sign[r_idx]*(g_r_t - g_rp_t))  # right segment of ring    
                    
                
                
        ## Store vortex distribution for this blade
        
        #sep  = filename.rfind('.')
        #VD_filename = filename[0:sep]+"_VD_blade"+str(B_idx)+filename[sep:]  
        #write_VD(rings,n_time_steps,n_radial_rings, VD_filename)
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
        for i in range(n_edges):
            # bound vortex scalar circulation
            f.write("\n"+str(ring_circulations[i])) 
                    
    return