## @ingroup Input_Output-VTK
# save_evaluation_points_vtk.py
# 
# Created:    Jun 2021, R. Erhard
# Modified: 
#           

## @ingroup Input_Output-VTK
def save_evaluation_points_vtk(points,filename="eval_pts.vtk",time_step=0):
    """
    Saves a SUAVE propeller wake as a VTK in legacy format.

    Inputs:
       points        Data structure of points to store information  [Unitless]  
          .XC
          .YC
          .ZC
          .point_data
          
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
    if len(points.XC) == 1:
        xp = points.XC[0]
        yp = points.YC[0]
        zp = points.ZC[0]
    else:
        xp = points.XC
        yp = points.YC
        zp = points.ZC        
    
    try:
        point_data = points.point_data
        wake=True
    except:
        print("No point data specified at evaluation points.")
        wake=False
    
    # Create file
    sep  = filename.rfind('.')
    file = filename[0:sep]+"."+str(time_step)+filename[sep:]
    with open(file, 'w') as f:
    
        #---------------------
        # Write header
        #---------------------
        header = ["# vtk DataFile Version 4.0" ,     # File version and identifier
                  "\nEvaluation points " ,           # Title
                  "\nASCII"              ,           # Data type
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
        
        #--------------------------        
        # Write Scalar Point Data:
        #--------------------------
        cell_data_header  = "\n\nPOINT_DATA "+str(n_pts)
        f.write(cell_data_header)      
        
        # First scalar value
        f.write("\nSCALARS i float")
        f.write("\nLOOKUP_TABLE default")   
        
        for i in range(len(xp)):
            f.write("\n"+str(i))      
            
        
        if wake:
            # Write each scalar value for all stored in data values
            for val in point_data.keys():
                try:
                    f.write("\nSCALARS "+val+" float")
                    f.write("\nLOOKUP_TABLE default")   
                    
                    for i in range(len(xp)):
                        f.write("\n"+str(point_data[val][i]))     
                except:
                    print("Error processing points.point_data."+val)
                       
    f.close()
        
        
    return