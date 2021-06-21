## @ingroup Time_Accurate-Simulations
# save_prop_vtk.py
# 
# Created:    Jun 2021, R. Erhard
# Modified: 
#           


def save_prop_vtk(vehicle, filename, Results, i_prop, Gprops):
    """
    Assumptions: 
         Quad cell structures for mesh
    """
    # Generate propeller point geometry
    prop     = vehicle.propulsors.battery_propeller.propeller
    n_blades = prop.number_of_blades
    
    
    for B_idx in range(n_blades):
        # Get geometry of blade for current propeller instance
        G = Gprops[i_prop][B_idx]
        
        sep  = filename.find('.')
        file = filename[0:sep]+"_blade"+str(B_idx)+filename[sep:]
        
        # Create file for each blade
        with open(file, 'w') as f:
            #---------------------
            # Write header
            #---------------------
            l1 = "# vtk DataFile Version 4.0"                     # File version and identifier
            l2 = "\nSUAVE Model of PROWIM Propeller Blade "       # Title 
            l3 = "\nASCII"                                        # Data type
            l4 = "\nDATASET UNSTRUCTURED_GRID"                    # Dataset structure / topology     
            
            header = [l1, l2, l3, l4]
            f.writelines(header)     
            
            # --------------------
            # Write Points
            # --------------------   
            n_r      = len(prop.chord_distribution)
            n_af     = Gprops.n_af
            
            n_vertices = (n_r)*(n_af)    # total number of node vertices per blade
            points_header = "\n\nPOINTS "+str(n_vertices) +" float"
            f.write(points_header)
            
            # Loop over all nodes
            for c_idx in range(n_af):
                for r_idx in range(n_r):
                    xp = round(G.X[r_idx,c_idx],4)
                    yp = round(G.Y[r_idx,c_idx],4)
                    zp = round(G.Z[r_idx,c_idx],4)
                        
                    new_point   = "\n"+str(xp)+" "+str(yp)+" "+str(zp)
                    f.write(new_point)  
                        
            #---------------------    
            # Write Cells:
            #---------------------
            cells_per_blade = n_af*(n_r-1)
            v_per_cell      = 4    # quad cells
            size            = cells_per_blade*(1+v_per_cell) # total number of integer values required to represent the list
            cell_header     = "\n\nCELLS "+str(cells_per_blade)+" "+str(size)
            f.write(cell_header)
            
            rlap=0
            for i in range(cells_per_blade): # loop over all nodes in blade
                if i==n_af-1+n_af*rlap:
                    # last airfoil face connects back to first node
                    b = i-(n_af-1)
                    c = i+1
                    rlap=rlap+1
                else:
                    b = i+1
                    c = i+n_af+1
                    
                a        = i
                d        = i+n_af
                new_cell = "\n4 "+str(a)+" "+str(b)+" "+str(c)+" "+str(d)
                
                f.write(new_cell)
                
                    
            
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
            f.write("\nSCALARS cl float 1")
            f.write("\nLOOKUP_TABLE default")  
            for i in range(cells_per_blade):
                new_idx = str(i)
                f.write("\n"+new_idx)
                
        f.close()
    
    
    return