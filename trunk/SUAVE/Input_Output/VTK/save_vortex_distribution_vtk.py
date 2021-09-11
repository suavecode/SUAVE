## @ingroup Input_Output-VTK
# save_evaluation_points_vtk.py
# 
# Created:    Jun 2021, R. Erhard
# Modified: 
#           
from SUAVE.Core import Data
import numpy as np

def save_vortex_distribution_vtk(vehicle,VD,wing_instance,filename, time_step):
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
    # Extract wake properties of the ith propeller
    symmetric = vehicle.wings[wing_instance.tag].symmetric
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
        
        sep  = filename.find('.')
        
        Lfile = filename[0:sep]+"_L"+"_t"+str(time_step)+filename[sep:]
        Rfile = filename[0:sep]+"_R"+"_t"+str(time_step)+filename[sep:]
                
        write_vortex_distribution_vtk(L,VD,Lfile)
        write_vortex_distribution_vtk(R,VD,Rfile)
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

        sep  = filename.find('.')
        file = filename[0:sep]+"_t"+str(time_step)+filename[sep:]
        
        write_vortex_distribution_vtk(wing,VD,file)
        
        
    return

def write_vortex_distribution_vtk(wing,VD,filename):
    # Create file
    with open(filename, 'w') as f:
        
        n_sw = VD.n_sw[0]
        n_cw = VD.n_cw[0]
        n_cp = n_sw*n_cw
        
        #---------------------
        # Write header
        #---------------------
        header = ["# vtk DataFile Version 4.0" ,     # File version and identifier
                  "\nEvaluation points " ,           # Title
                  "\nASCII"              ,           # Data type
                  "\nDATASET UNSTRUCTURED_GRID"             ] # Dataset structure / topology
        f.writelines(header)   
        
        # --------------------
        # Write Points
        # --------------------   
        pts_per_horseshoe = 4
        n_pts = n_cp*pts_per_horseshoe  # total number of node vertices
        points_header = "\n\nPOINTS "+str(n_pts) +" float"
        f.write(points_header)
        
        points_array = Data()
        cells_array  = Data()
        for s in range(n_sw):
            for c in range(n_cw):
                i = c + s*(n_cw)
                if c==n_cw-1:
                    # last row uses trailing edge points instead of A2 and B2
                    p0 = np.array([wing.XA_TE[i], wing.YA_TE[i], wing.ZA_TE[i]])
                    p1 = np.array([wing.XA_TE[i], wing.YA_TE[i], wing.ZA_TE[i]])
                    p2 = np.array([wing.XAH[i], wing.YAH[i], wing.ZAH[i]])
                    p3 = np.array([wing.XBH[i], wing.YBH[i], wing.ZBH[i]])
                    p4 = np.array([wing.XB_TE[i], wing.YB_TE[i], wing.ZB_TE[i]])
                    p5 = np.array([wing.XB_TE[i], wing.YB_TE[i], wing.ZB_TE[i]])
                elif s==0:
                    #-------------------------
                    # Leading-edge panels
                    #-------------------------
                    # bound vortices and first part of horseshoe of first panel (equal strength)
                    p1 = np.array([wing.XAH[i+1], wing.YAH[i+1], wing.ZAH[i+1]])   # bound vortex of next chordwise panel
                    p2 = np.array([wing.XAH[i], wing.YAH[i], wing.ZAH[i]])         # bound vortex of current panel
                    p3 = np.array([wing.XBH[i], wing.YBH[i], wing.ZBH[i]])         # bound vortex of current panel
                    p4 = np.array([wing.XBH[i+1], wing.YBH[i+1], wing.ZBH[i+1]])   # bound vortex of next chordwise panel
                    
                    points_array.append(["\n"+str(p1[0])+" "+str(p1[1])+" "+str(p1[2])])
                    points_array.append(["\n"+str(p2[0])+" "+str(p2[1])+" "+str(p2[2])])
                    points_array.append(["\n"+str(p3[0])+" "+str(p3[1])+" "+str(p3[2])])
                    points_array.append(["\n"+str(p4[0])+" "+str(p4[1])+" "+str(p4[2])])
                else:
                    # for each horseshoe, draw the line from TE --> A2 --> AH --> BH --> B2
                    p0 = np.array([wing.XA_TE[i], wing.YA_TE[i], wing.ZA_TE[i]])
                    p1 = np.array([wing.XA2[i], wing.YA2[i], wing.ZA2[i]])
                    p2 = np.array([wing.XAH[i], wing.YAH[i], wing.ZAH[i]])
                    p3 = np.array([wing.XBH[i], wing.YBH[i], wing.ZBH[i]])
                    p4 = np.array([wing.XB2[i], wing.YB2[i], wing.ZB2[i]])   
                    p5 = np.array([wing.XB_TE[i], wing.YB_TE[i], wing.ZB_TE[i]])                 
                
                #f.write("\n"+str(p0[0])+" "+str(p0[1])+" "+str(p0[2]))
                f.write("\n"+str(p1[0])+" "+str(p1[1])+" "+str(p1[2]))
                f.write("\n"+str(p2[0])+" "+str(p2[1])+" "+str(p2[2]))
                f.write("\n"+str(p3[0])+" "+str(p3[1])+" "+str(p3[2]))
                f.write("\n"+str(p4[0])+" "+str(p4[1])+" "+str(p4[2]))
                #f.write("\n"+str(p5[0])+" "+str(p5[1])+" "+str(p5[2]))
            
        
        #---------------------    
        # Write Cells:
        #---------------------     
        n          = n_cp
        v_per_cell = pts_per_horseshoe
        size       = n*(1+v_per_cell)
        
        cell_header  = "\n\nCELLS "+str(n_cp)+" "+str(size)
        f.write(cell_header)
        for i in range(n_cp):
            base_node = i*pts_per_horseshoe
            new_poly_line = "\n4 "+str(base_node)+" "+str(base_node+1)+" "+str(base_node+2)+" "+str(base_node+3)#+" "+str(base_node+4)+" "+str(base_node+5)
            f.write(new_poly_line )
           
        #---------------------        
        # Write Cell Types:
        #---------------------
        cell_type_header  = "\n\nCELL_TYPES "+str(n_cp)
        f.write(cell_type_header)        
        for i in range(n_cp):
            f.write("\n4")
        
        
        #--------------------------        
        # Write VTK Poly Line Data:
        #--------------------------
        cell_data_header  = "\n\nCELL_DATA "+str(n_cp)
        f.write(cell_data_header)      
        
        # First scalar value
        f.write("\nSCALARS i float 1")
        f.write("\nLOOKUP_TABLE default")   
        
        for i in range(n_cp):
            f.write("\n"+str(i))          
            
        # Second scalar value
        f.write("\nSCALARS gamma float 1")
        f.write("\nLOOKUP_TABLE default")   
        
        for i in range(n_cp):
            f.write("\n"+str(round(VD.gamma[0,i],4)))                   
                       
    f.close()
    return