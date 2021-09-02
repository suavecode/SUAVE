## @ingroup Input_Output-VTK
# save_evaluation_points_vtk.py
# 
# Created:    Jun 2021, R. Erhard
# Modified: 
#           

def save_evaluation_points_vtk(points,filename):
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
    xp = points.XC
    yp = points.YC
    zp = points.ZC
    
    # Create file
    with open(filename, 'w') as f:
    
        #---------------------
        # Write header
        #---------------------
        header = ["# vtk DataFile Version 4.0"              , # File version and identifier
                  "\nEvaluation points " , # Title
                  "\nASCII"                                 , # Data type
                  "\nDATASET POLYDATA"             ] # Dataset structure / topology
        f.writelines(header)   
        
        # --------------------
        # Write Points
        # --------------------   
        n_pts = len(xp)    # total number of node vertices
        points_header = "\n\nPOINTS "+str(n_pts) +" float"
        f.write(points_header)
        
        for i in range(len(xp)):       
            new_point = "\n"+str(xp[i])+" "+str(yp[i])+" "+str(zp[i])
            f.write(new_point)                
        ##---------------------    
        ## Write Cells:
        ##---------------------
        
        #cell_header     = "\n\nCELLS "+str(n_pts)+" "+str(n_pts)
        #f.write(cell_header)
        
        #for i in range(len(xp)):
            #new_cell = "\n1 "+str(i)
            #f.write(new_cell)
        
        ##---------------------        
        ## Write Cell Types:
        ##---------------------
        #cell_type_header  = "\n\nCELL_TYPES "+str(n_pts)
        #f.write(cell_type_header)        
        #for i in range(n_pts):
            #f.write("\n1")    
            
        #--------------------------        
        # Write Scalar Cell Data:
        #--------------------------
        cell_data_header  = "\n\nPOINT_DATA "+str(n_pts)
        f.write(cell_data_header)      
        
        # First scalar value
        f.write("\nSCALARS i float")
        f.write("\nLOOKUP_TABLE default")   
        
        for i in range(len(xp)):
            f.write("\n"+str(i))           
                       
    f.close()
        
        
    return